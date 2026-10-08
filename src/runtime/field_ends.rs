//! The decoder's account of how it ended each string field of a JSON answer
//! (findit-studio/application#235): whether the model closed the string or
//! the grammar closed it at the field's `maxLength`.
//!
//! [`ConstrainedSampler`](super::sampler::ConstrainedSampler) hands every
//! token it commits to a [`FieldTracker`], together with the mask the token
//! was drawn under. The tracker follows the JSON structure of the committed
//! bytes far enough to record each member of the answer's top-level object:
//! its key, and — when its value is a string — the string's lexeme exactly as
//! committed, where it sits in the committed bytes, and who closed it
//! ([`ClosedBy`]), read off that mask. [`field_ends`] then binds each record
//! to the string the answer carries, as llmtask's [`FieldEnd`].
//!
//! # Why the tracker is exact
//!
//! llguidance exposes no JSON path for the string being generated: its
//! `Constraint` hands out the token mask, the commit result and the stop
//! state, and its parser works on lexemes compiled from the schema, not on
//! the schema's properties. The tracker reads the bytes the matcher itself
//! consumed — each committed token's bytes from the matcher's own `TokTrie` —
//! and JSON's structure (inside or outside a string, escapes, nesting, a key
//! or a value) is fixed by those bytes alone. The matcher admits only bytes
//! that continue a JSON text the schema accepts, so the tracker never meets a
//! byte it has to guess about.
//!
//! # Who closed a string
//!
//! A string closes at a byte of the drawn token — its first, or one past
//! content the token carries, or one past the string's opening quote when the
//! token opens it too. Every such close is read against the whole mask the
//! token was drawn under, by the bytes each allowed token would have
//! committed from the state the step began in:
//!
//! - **the model's** when an allowed token commits the drawn token's bytes up
//!   to the close and then continues the string instead: the model could have
//!   written on, and chose the close;
//! - **the grammar's** when every allowed token closes that same string: none
//!   would have left it open, so no choice the model had kept it going;
//! - **unproven** otherwise — say the drawn token writes content and closes,
//!   and the mask holds a shorter token that stops inside the string but none
//!   that writes past the drawn token's content: whether the grammar would
//!   have taken more after that content is not in the mask. Below the cap
//!   such a string has no account, rather than one the mask cannot back.
//!
//! When llguidance forces bytes it can narrow the mask to the single token
//! that starts their canonical tokenization (`TokenParser::compute_mask`);
//! that token may open a string, write its content and close it in one —
//! `""` for a `maxLength` of 0 — and as the only allowed token it closes the
//! string on every road the mask leaves, so the close is the grammar's.
//!
//! # The account: length first, then the mask
//!
//! [`field_ends`] decides each account by the string's length first, against
//! the cap the task declares for its member ([`llmtask::Task::field_caps`]).
//! lfm derives no cap from the schema: the task that wrote it is the one
//! authority on its caps. A string that closed holding exactly its declared
//! cap was bound by it, whichever token carried the quote — a lone `"`, a `.`
//! before a forced close, or `."` one character short: the model could not
//! have written past it, so it is `cap`, whatever the mask shows. Only below
//! the cap — or for a member that declares none — does the mask's reading
//! decide: the model's close is `model`; a close every allowed token made
//! there (an `enum`, a `const`, a `pattern`, or a cap the task left
//! undeclared) has no account; an unproven one has none either.
//!
//! # One lexeme, bound on its own
//!
//! An account is bound to its member's string lexeme alone: the bytes the
//! answer carries where the matcher committed the string, decoded by
//! serde_json as one JSON string. Nothing else in the answer has to decode:
//! a schema-valid member that serde_json cannot materialize as a `Value` — a
//! number outside the `f64` range (`1e400`), a value nested past its
//! recursion limit — costs no other member its account. A key written twice
//! is read as serde_json reads it, by its last occurrence.
//!
//! # No cut token
//!
//! An account never names a cut token ([`FieldEnd::with_cut`]): a string's
//! bytes are always whole characters when it closes. llguidance (1.7.6 and
//! 1.8.0, by source) compiles a string with a `maxLength` to `(?s:.{0,N})`
//! under its JSON quoting (`gen_json_string`), counting characters, and its
//! lexer builds `.` from complete UTF-8 sequences, so a closing quote is never
//! admitted after a partial one; and `generate` detokenizes the whole answer
//! in one call, which for the bundled byte-level tokenizer joins every token's
//! bytes before it decodes them. A field's text therefore never ends in a
//! U+FFFD left by a token cut at the cap.

use std::collections::BTreeMap;

use llguidance::toktrie::{SimpleVob, TokTrie};
use llmtask::{FieldEnd, FieldEnds};
use smol_str::SmolStr;

use llmtask::FieldCaps;

/// One member of the answer's top-level object, as the decode wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Member {
  /// The member's key, JSON-decoded.
  pub(crate) field: SmolStr,
  /// Its value when that is a string; `None` for any other value.
  pub(crate) string: Option<MemberString>,
}

/// The string value of a top-level member: where the decode wrote it, its
/// lexeme, and who closed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemberString {
  /// The offset of its opening quote in the committed bytes.
  pub(crate) at: usize,
  /// Its lexeme as committed, opening quote to closing quote.
  pub(crate) lexeme: Vec<u8>,
  /// Who closed it, as the mask its closing token was drawn under shows.
  pub(crate) closed_by: ClosedBy,
}

/// Who closed a string, read against the mask the token that closes it was
/// drawn under (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClosedBy {
  /// An allowed token would have written on where the drawn token closes it.
  Model,
  /// Every allowed token closes it.
  Grammar,
  /// The mask shows neither.
  Unproven,
}

/// The tokens one step's mask allowed, by their bytes.
pub(crate) trait Allowed {
  /// Whether `test` holds for some allowed token.
  fn any(&self, test: impl FnMut(&[u8]) -> bool) -> bool;

  /// Whether `test` holds for every allowed token.
  fn all(&self, mut test: impl FnMut(&[u8]) -> bool) -> bool {
    !self.any(|token| !test(token))
  }
}

/// The tokens a step's mask allowed, read off the matcher's token trie.
pub(crate) struct Masked<'a> {
  pub(crate) mask: &'a SimpleVob,
  pub(crate) trie: &'a TokTrie,
}

impl Allowed for Masked<'_> {
  fn any(&self, mut test: impl FnMut(&[u8]) -> bool) -> bool {
    self.mask.iter().any(|token| test(self.trie.token(token)))
  }
}

/// Follows the JSON structure of the committed bytes far enough to record
/// each member of the top-level object.
#[derive(Debug, Default)]
pub(crate) struct FieldTracker {
  cursor: Cursor,
  /// How many bytes have been committed.
  offset: usize,
  /// The top-level key being read, as written (escapes included).
  key: Vec<u8>,
  /// The decoded key of the member whose value is being read, until the
  /// member is recorded.
  member: Option<SmolStr>,
  /// Where the member string being read opened, and its lexeme so far.
  string_at: usize,
  lexeme: Vec<u8>,
  members: Vec<Member>,
}

impl FieldTracker {
  /// Nothing committed yet.
  // Its one caller, `Engine::run`, is compiled only with `decoders` on.
  #[cfg_attr(not(feature = "decoders"), allow(dead_code))]
  pub(crate) fn new() -> Self {
    Self::default()
  }

  /// The members of the top-level object recorded so far, in the order the
  /// answer wrote them; a member is recorded once its value ends.
  #[cfg_attr(not(feature = "decoders"), allow(dead_code))]
  pub(crate) fn members(&self) -> &[Member] {
    &self.members
  }

  /// Follows one committed token, `token`, drawn under a mask that allowed
  /// `allowed`: every top-level member string it closes is recorded with who
  /// closed it ([`ClosedBy`]), read against that whole mask from the state
  /// the step began in.
  pub(crate) fn commit(&mut self, token: &[u8], allowed: &(impl Allowed + ?Sized)) {
    let before = self.cursor.clone();
    for (at, &byte) in token.iter().enumerate() {
      match self.cursor.step(byte) {
        Event::Opens(Role::Key) => self.key.clear(),
        Event::Content(Role::Key) => self.key.push(byte),
        Event::Closes(Role::Key) => self.member = decode_key(&self.key),
        Event::Opens(Role::Member) => {
          self.string_at = self.offset;
          self.lexeme.clear();
          self.lexeme.push(byte);
        }
        Event::Content(Role::Member) => self.lexeme.push(byte),
        Event::Closes(Role::Member) => {
          self.lexeme.push(byte);
          let lexeme = std::mem::take(&mut self.lexeme);
          if let Some(field) = self.member.take() {
            self.members.push(Member {
              field,
              string: Some(MemberString {
                at: self.string_at,
                lexeme,
                closed_by: closed_by(&before, token, at, allowed),
              }),
            });
          }
        }
        Event::MemberEnds => {
          if let Some(field) = self.member.take() {
            self.members.push(Member {
              field,
              string: None,
            });
          }
        }
        Event::Opens(Role::Other)
        | Event::Content(Role::Other)
        | Event::Closes(Role::Other)
        | Event::Outside => {}
      }
      self.offset += 1;
    }
  }
}

/// Where the committed bytes stand in the JSON text.
#[derive(Debug, Clone, Default)]
struct Cursor {
  /// The containers open around the next byte, outermost first.
  open: Vec<Container>,
  /// The string the next byte is inside, if any.
  string: Option<OpenString>,
  /// How many keys of the top-level object have opened: the member a
  /// top-level string value belongs to is the one this many keys in.
  keys: usize,
  /// The top-level value has closed: no later byte belongs to it.
  done: bool,
}

#[derive(Debug, Clone, Copy)]
enum Container {
  /// An object; `key_next` while the next string in it is a key.
  Object {
    key_next: bool,
  },
  Array,
}

#[derive(Debug, Clone, Copy)]
struct OpenString {
  role: Role,
  /// The previous byte was an unescaped backslash, so this one is escaped.
  escaped: bool,
}

/// What a string is to the top-level object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
  /// One of its keys.
  Key,
  /// The value of the member being read.
  Member,
  /// Any other string: a nested key or value, an array's element.
  Other,
}

/// What one byte does to the JSON text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
  /// Nothing the tracker records: structure, whitespace, a number's or a
  /// literal's bytes, or a byte past the top-level value.
  Outside,
  /// It opens a string: its opening quote.
  Opens(Role),
  /// It is a string's content, escapes included.
  Content(Role),
  /// It closes a string: its closing quote.
  Closes(Role),
  /// It ends a member of the top-level object: the `,` after it, or the `}`
  /// that closes the object.
  MemberEnds,
}

impl Cursor {
  fn step(&mut self, byte: u8) -> Event {
    if self.done {
      return Event::Outside;
    }
    if let Some(string) = &mut self.string {
      let role = string.role;
      if !string.escaped && byte == b'"' {
        self.string = None;
        return Event::Closes(role);
      }
      string.escaped = !string.escaped && byte == b'\\';
      return Event::Content(role);
    }
    let in_top_level_object =
      self.open.len() == 1 && matches!(self.open.last(), Some(Container::Object { .. }));
    match byte {
      b'{' => self.open.push(Container::Object { key_next: true }),
      b'[' => self.open.push(Container::Array),
      b'}' | b']' => {
        self.open.pop();
        self.done = self.open.is_empty();
        if in_top_level_object {
          return Event::MemberEnds;
        }
      }
      b',' => {
        if let Some(Container::Object { key_next }) = self.open.last_mut() {
          *key_next = true;
        }
        if in_top_level_object {
          return Event::MemberEnds;
        }
      }
      b'"' => {
        let role = self.open_role();
        self.string = Some(OpenString {
          role,
          escaped: false,
        });
        return Event::Opens(role);
      }
      // `:`, whitespace, and the bytes of a number, `true`, `false` or `null`.
      _ => {}
    }
    Event::Outside
  }

  /// The role of a string opening at the next byte.
  fn open_role(&mut self) -> Role {
    let top_level = self.open.len() == 1;
    match self.open.last_mut() {
      Some(Container::Object { key_next }) if *key_next => {
        *key_next = false;
        if top_level {
          self.keys += 1;
          Role::Key
        } else {
          Role::Other
        }
      }
      Some(Container::Object { .. }) if top_level => Role::Member,
      _ => Role::Other,
    }
  }
}

impl Cursor {
  /// Whether `bytes`, read on from this state, close the string value of the
  /// top-level member `key` keys in.
  fn closes_member(mut self, bytes: &[u8], key: usize) -> bool {
    bytes
      .iter()
      .any(|&byte| self.step(byte) == Event::Closes(Role::Member) && self.keys == key)
  }
}

/// Who closed the member string that `token`, drawn from the state `before`
/// under a mask that allowed `allowed`, closes at its byte `close`.
fn closed_by(
  before: &Cursor,
  token: &[u8],
  close: usize,
  allowed: &(impl Allowed + ?Sized),
) -> ClosedBy {
  // After `written` the string is open and unescaped: `token[close]` closes it.
  let written = &token[..close];
  let writes_on = |other: &[u8]| {
    other.starts_with(written)
      && other
        .get(close)
        .is_some_and(|&next| next != b'"' && next != TokTrie::SPECIAL_TOKEN_MARKER)
  };
  if allowed.any(writes_on) {
    return ClosedBy::Model;
  }
  let mut at_close = before.clone();
  for &byte in written {
    at_close.step(byte);
  }
  let key = at_close.keys;
  if allowed.all(|other| before.clone().closes_member(other, key)) {
    ClosedBy::Grammar
  } else {
    ClosedBy::Unproven
  }
}

/// A key's bytes as written between its quotes, JSON-decoded.
fn decode_key(written: &[u8]) -> Option<SmolStr> {
  let mut quoted = Vec::with_capacity(written.len() + 2);
  quoted.push(b'"');
  quoted.extend_from_slice(written);
  quoted.push(b'"');
  serde_json::from_slice::<String>(&quoted)
    .ok()
    .map(SmolStr::from)
}

/// The decoder's account of each top-level string member of `raw` that
/// `members` records, for the task's `parse_ended`.
///
/// Each account is bound to its member's string lexeme alone, as `raw`
/// carries it at the offset the matcher committed it — byte for byte the
/// lexeme the tracker recorded — and decoded by serde_json as one JSON
/// string: JSON-decoded and untrimmed, the string the task's own parse reads
/// from the same text. Nothing else in `raw` has to decode.
///
/// The account is decided by the string's length first, then by the mask:
///
/// 1. a string holding exactly its member's cap — the `maxLength` the task
///    declares for it in `caps` ([`llmtask::Task::field_caps`]), in Unicode
///    scalar values — was bound by the cap, whichever token carried the
///    quote: the model could not have written past it, so it is
///    [`FieldEnd::cap`];
/// 2. below the cap, or for a member with no declared cap, a close the mask
///    shows was the model's ([`ClosedBy::Model`]) is [`FieldEnd::model`];
/// 3. otherwise — a close every allowed token made (an `enum`, a `const`, a
///    `pattern`, or a cap the task did not declare), or one the mask cannot
///    attribute — there is no account.
///
/// A key written more than once is read as serde_json reads it, by its last
/// occurrence: that occurrence's string binds, and when its value is not a
/// string the key has no account. A member also gets no account when `raw`
/// does not carry its lexeme where the matcher committed it, or when the
/// lexeme does not decode.
// Its one caller, `Engine::run`, is compiled only with `decoders` on.
#[cfg_attr(not(feature = "decoders"), allow(dead_code))]
pub(crate) fn field_ends(raw: &str, members: &[Member], caps: &FieldCaps) -> FieldEnds {
  let mut last: BTreeMap<&str, Option<FieldEnd>> = BTreeMap::new();
  for member in members {
    let field = member.field.as_str();
    let end = member
      .string
      .as_ref()
      .and_then(|string| account(raw, string, caps.get(field)));
    last.insert(field, end);
  }
  let mut ends = FieldEnds::new();
  for (field, end) in last {
    if let Some(end) = end {
      ends.insert(field, end);
    }
  }
  ends
}

/// The account of one member string, bound to its lexeme as `raw` carries
/// it; `cap` is the `maxLength` the task declares for the member.
fn account(raw: &str, string: &MemberString, cap: Option<usize>) -> Option<FieldEnd> {
  let lexeme = raw.get(string.at..string.at.checked_add(string.lexeme.len())?)?;
  if lexeme.as_bytes() != string.lexeme.as_slice() {
    return None;
  }
  let text: String = serde_json::from_str(lexeme).ok()?;
  // The length decides first: a string that closed holding exactly its cap
  // was bound by it, whichever token carried the quote.
  if cap == Some(text.chars().count()) {
    return Some(FieldEnd::cap(&text));
  }
  match string.closed_by {
    ClosedBy::Model => Some(FieldEnd::model(&text)),
    ClosedBy::Grammar | ClosedBy::Unproven => None,
  }
}

/// Driving the real llguidance matcher over a scripted answer, with the
/// bundled tokenizer: no model runs.
#[cfg(test)]
pub(crate) mod testing {
  use std::{path::PathBuf, sync::OnceLock};

  use llguidance::{Constraint, ParserFactory, api::TopLevelGrammar, toktrie::TokTrie};

  /// The bundled `tokenizer.json`.
  pub(crate) fn bundled_tokenizer_json() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
      let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/tokenizer.json");
      std::fs::read(path).expect("the bundled tokenizer.json reads")
    })
  }

  /// The parser factory `Engine` builds, over the bundled tokenizer.
  pub(crate) fn factory() -> &'static ParserFactory {
    static FACTORY: OnceLock<ParserFactory> = OnceLock::new();
    FACTORY.get_or_init(|| {
      let tok_env = toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(bundled_tokenizer_json())
        .expect("the bundled tokenizer loads")
        .into_tok_env(None)
        .expect("its token trie builds");
      ParserFactory::new_simple(&tok_env).expect("the factory builds")
    })
  }

  /// The matcher's token trie.
  pub(crate) fn trie() -> &'static TokTrie {
    factory().tok_env().tok_trie()
  }

  /// The constraint `Engine::run` builds for a JSON Schema.
  pub(crate) fn constraint(schema: &serde_json::Value) -> Constraint {
    let grammar = TopLevelGrammar::from_json_schema(schema.clone());
    Constraint::new(
      factory()
        .create_parser(grammar)
        .expect("the schema compiles"),
    )
  }

  /// A text step.
  pub(crate) fn text(text: &str) -> Step {
    Step::Text(text.into())
  }

  /// `count` characters of a caption that opens with a space, so a trimmed
  /// account would not be the string the answer carries.
  pub(crate) fn caption(count: usize) -> String {
    let text: String = " A grey cat sleeps on a sunlit windowsill beside a potted fern, while \
                        rain streaks the glass and a kettle steams on the stove behind"
      .chars()
      .take(count)
      .collect();
    assert_eq!(text.chars().count(), count, "the caption is long enough");
    text
  }

  /// One step of a scripted answer.
  #[derive(Debug, Clone)]
  pub(crate) enum Step {
    /// This text, drawn as the longest token the matcher allows at each
    /// step that the text still starts with — so a token never spans two
    /// steps.
    Text(String),
    /// Exactly this token.
    Token(u32),
  }

  /// A scripted answer, played one token at a time: the logits it hands a
  /// greedy sampler make its next token the pick wherever the matcher
  /// allows it.
  #[derive(Debug, Clone)]
  pub(crate) struct Script {
    steps: Vec<Step>,
    /// The step being played, and how many of its text's bytes are drawn.
    at: usize,
    drawn: usize,
  }

  impl Script {
    pub(crate) fn new(steps: Vec<Step>) -> Self {
      Self {
        steps,
        at: 0,
        drawn: 0,
      }
    }

    /// Whether every step has been drawn.
    pub(crate) fn finished(&self) -> bool {
      self.at == self.steps.len()
    }

    /// The logits for the next draw, over `vocab` tokens: a text step scores
    /// each token its text continues with by its length in bytes, a token
    /// step scores its token, and every other token is far below. Past the
    /// script, the matcher's end-of-sequence token is the pick.
    pub(crate) fn logits(&self, vocab: usize) -> Vec<f32> {
      let mut logits = vec![-1_000.0_f32; vocab];
      match self.steps.get(self.at) {
        Some(Step::Text(text)) => {
          let rest = &text.as_bytes()[self.drawn..];
          for (token, logit) in logits.iter_mut().enumerate() {
            let bytes = trie().token(token as u32);
            if !bytes.is_empty()
              && bytes[0] != TokTrie::SPECIAL_TOKEN_MARKER
              && rest.starts_with(bytes)
            {
              *logit = bytes.len() as f32;
            }
          }
        }
        Some(Step::Token(token)) => logits[*token as usize] = 1.0,
        None => logits[trie().eos_token() as usize] = 1.0,
      }
      logits
    }

    /// Records that `token` was drawn, which must be the script's next.
    pub(crate) fn drew(&mut self, token: u32) {
      let bytes = trie().token(token);
      match self.steps.get(self.at) {
        Some(Step::Text(text)) => {
          let rest = &text.as_bytes()[self.drawn..];
          assert!(
            rest.starts_with(bytes),
            "drew {:?} where the script goes on with {:?}",
            trie().token_str(token),
            String::from_utf8_lossy(rest)
          );
          self.drawn += bytes.len();
          if self.drawn == text.len() {
            self.at += 1;
            self.drawn = 0;
          }
        }
        Some(Step::Token(expected)) => {
          assert_eq!(
            token,
            *expected,
            "drew {:?} where the script holds {:?}",
            trie().token_str(token),
            trie().token_str(*expected)
          );
          self.at += 1;
        }
        None => assert_eq!(token, trie().eos_token(), "drew past the script"),
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use llmtask::{
    DescriptionEnd, Task,
    image_analysis::{Extension, ImageAnalysisTask},
  };
  use serde_json::{Value, json};

  use super::{
    testing::{Script, Step, caption, constraint, text, trie},
    *,
  };
  use crate::{
    options::RequestOptions,
    runtime::sampler::{ConstrainedSampler, SampleResult, Sampler},
  };

  /// The caps a task declares, as `(field, maxLength)` pairs.
  fn declared(caps: &[(&str, usize)]) -> FieldCaps {
    let mut declared = FieldCaps::new();
    for &(field, cap) in caps {
      declared.insert(field, cap);
    }
    declared
  }

  // ===== the tracker, byte by byte =====

  impl Allowed for [&[u8]] {
    fn any(&self, test: impl FnMut(&[u8]) -> bool) -> bool {
      self.iter().copied().any(test)
    }
  }

  /// The members a tracker records over `tokens`, each a committed token's
  /// bytes drawn as the only token its mask allowed: every close is the
  /// grammar's.
  fn follow(tokens: &[&[u8]]) -> Vec<Member> {
    let mut tracker = FieldTracker::new();
    for token in tokens {
      tracker.commit(token, &[*token][..]);
    }
    tracker.members().to_vec()
  }

  /// The members a tracker records over `raw` committed as one token, each
  /// string member then accounted closed by `by`, in order.
  fn recorded(raw: &str, by: &[ClosedBy]) -> Vec<Member> {
    let mut members = follow(&[raw.as_bytes()]);
    let mut by = by.iter();
    for member in &mut members {
      if let Some(string) = &mut member.string {
        string.closed_by = *by.next().expect("an account per string member");
      }
    }
    assert!(by.next().is_none(), "a string member per account");
    members
  }

  /// The member `field` of `answer` whose string value is `lexeme`, found
  /// where `answer` writes it.
  fn string(answer: &[u8], field: &str, lexeme: &str, closed_by: ClosedBy) -> Member {
    let at = answer
      .windows(lexeme.len())
      .position(|window| window == lexeme.as_bytes())
      .expect("the answer writes the lexeme");
    Member {
      field: field.into(),
      string: Some(MemberString {
        at,
        lexeme: lexeme.as_bytes().to_vec(),
        closed_by,
      }),
    }
  }

  /// The member `field` whose value is not a string.
  fn non_string(field: &str) -> Member {
    Member {
      field: field.into(),
      string: None,
    }
  }

  /// Each string member's key and who closed it.
  fn closes(members: &[Member]) -> Vec<(&str, ClosedBy)> {
    members
      .iter()
      .filter_map(|member| Some((member.field.as_str(), member.string.as_ref()?.closed_by)))
      .collect()
  }

  /// LAW: **every member of the top-level object is recorded once, in
  /// order** — a string value with its lexeme exactly as written and where it
  /// sits, any other value as no string — and nothing else is: not an
  /// array's elements, not a nested object's keys or values, however the
  /// bytes fall into tokens and whatever whitespace sits between them.
  #[test]
  fn each_top_level_member_is_recorded_once_in_order() {
    let answer: &[u8] = br#"{ "scene" : "kitchen", "tags":["cat","rug"], "meta":{"note":"x","list":["y"]}, "description":"A \"cat\".", "n": 1.5e3, "ok": true }"#;
    let expected = [
      string(answer, "scene", r#""kitchen""#, ClosedBy::Grammar),
      non_string("tags"),
      non_string("meta"),
      string(answer, "description", r#""A \"cat\".""#, ClosedBy::Grammar),
      non_string("n"),
      non_string("ok"),
    ];
    assert_eq!(follow(&[answer]), expected, "one token");
    for size in 1..9 {
      let tokens: Vec<&[u8]> = answer.chunks(size).collect();
      assert_eq!(follow(&tokens), expected, "tokens of {size} bytes");
    }
  }

  /// Who closed the member string `token` closes, drawn after `prefix`
  /// under a mask that allowed `allowed`.
  fn by(prefix: &[u8], token: &[u8], allowed: &[&[u8]]) -> ClosedBy {
    let mut tracker = FieldTracker::new();
    tracker.commit(prefix, &[prefix][..]);
    let before = tracker.members().len();
    tracker.commit(token, allowed);
    let closed = &tracker.members()[before..];
    assert_eq!(closed.len(), 1, "{token:?} closes one member string");
    closed[0]
      .string
      .as_ref()
      .expect("a string member")
      .closed_by
  }

  /// LAW (Codex R1, [medium]): **every close in a drawn token is read against
  /// the whole mask it was drawn under**, wherever in the token it falls.
  /// The model closed the string when an allowed token writes the drawn
  /// token's bytes up to the close and then writes on; the grammar did when
  /// every allowed token closes that same string; and when the mask shows
  /// neither, the close is unproven.
  #[test]
  fn every_close_in_a_token_is_read_against_its_mask() {
    // A close at the token's first byte, the string already open.
    let open: &[u8] = br#"{"description":"A cat"#;
    let (close, comma, s) = (&br#"""#[..], &br#"","#[..], &b"s"[..]);
    assert_eq!(by(open, comma, &[comma, close]), ClosedBy::Grammar);
    assert_eq!(by(open, comma, &[comma, s]), ClosedBy::Model);
    // A close past content the token writes.
    let s_close: &[u8] = br#"s""#;
    assert_eq!(by(open, s_close, &[s_close]), ClosedBy::Grammar);
    assert_eq!(by(open, s_close, &[s_close, b"s."]), ClosedBy::Model);
    assert_eq!(
      by(open, s_close, &[s_close, s]),
      ClosedBy::Unproven,
      "`s` stops inside the string; nothing writes past it"
    );
    // A string the token opens and closes.
    let key: &[u8] = br#"{"description":"#;
    let empty: &[u8] = br#""""#;
    assert_eq!(
      by(key, empty, &[empty]),
      ClosedBy::Grammar,
      "a forced `\"\"`"
    );
    assert_eq!(by(key, empty, &[empty, br#""A"#]), ClosedBy::Model);
    assert_eq!(
      by(key, empty, &[empty, br#"""#]),
      ClosedBy::Unproven,
      "`\"` opens it and stops"
    );
    assert_eq!(
      by(key, empty, &[empty, b" "]),
      ClosedBy::Unproven,
      "a space does not reach the string"
    );
  }

  /// LAW (Codex R1, [medium]): **a later close in the same token is read
  /// against the same mask**: the token that closes the description and then
  /// opens and closes `scene` was the only one allowed, so both closes are the
  /// grammar's; with a lone quote allowed too — it closes the description and
  /// stops short of `scene` — the description's close is still the grammar's
  /// and `scene`'s is unproven.
  #[test]
  fn a_later_close_in_a_token_is_read_against_the_same_mask() {
    let open: &[u8] = br#"{"description":"A cat"#;
    let token: &[u8] = br#"","scene":"x""#;
    let close: &[u8] = br#"""#;
    for (allowed, scene) in [
      (vec![token], ClosedBy::Grammar),
      (vec![token, close], ClosedBy::Unproven),
    ] {
      let mut tracker = FieldTracker::new();
      tracker.commit(open, &[open][..]);
      tracker.commit(token, &allowed[..]);
      assert_eq!(
        closes(tracker.members()),
        [("description", ClosedBy::Grammar), ("scene", scene)]
      );
    }
  }

  /// LAW: **an escaped quote or backslash never closes a string**, wherever
  /// the token boundary falls inside the escape; a key is named as JSON
  /// decodes it, and a lexeme is kept as written.
  #[test]
  fn escapes_never_close_a_string_and_keys_are_decoded() {
    let answer: &[u8] = br#"{"d\u0065scription":"say \"hi\" \\","sc\"ene":"\\\""}"#;
    let expected = [
      string(
        answer,
        "description",
        r#""say \"hi\" \\""#,
        ClosedBy::Grammar,
      ),
      string(answer, "sc\"ene", r#""\\\"""#, ClosedBy::Grammar),
    ];
    for size in 1..12 {
      let tokens: Vec<&[u8]> = answer.chunks(size).collect();
      assert_eq!(follow(&tokens), expected, "tokens of {size} bytes");
    }
  }

  /// LAW: **only a member of a top-level object is a field**: the strings of
  /// a top-level array or a top-level string are no field's, and nothing
  /// after the top-level value counts.
  #[test]
  fn only_a_member_of_a_top_level_object_is_a_field() {
    assert!(follow(&[br#"["a","b"]"#]).is_empty());
    assert!(follow(&[br#""abc""#]).is_empty());
    let tokens: [&[u8]; 3] = [br#"{"a":"x"}"#, b" ", br#""b":"y""#];
    assert_eq!(closes(&follow(&tokens)), [("a", ClosedBy::Grammar)]);
  }

  // ===== binding the members to the answer =====

  /// LAW: **an account holds the field exactly as serde_json decodes its
  /// lexeme** — escapes decoded, untrimmed — the model's close as `model`,
  /// the grammar's at the declared cap as `cap`.
  #[test]
  fn an_account_holds_the_decoded_untrimmed_field() {
    let raw = r#"{"description":"  A \"cat\"","scene":" kitchen "}"#;
    let members = recorded(raw, &[ClosedBy::Grammar, ClosedBy::Model]);
    let ends = field_ends(raw, &members, &declared(&[("description", 9)]));
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap("  A \"cat\"")));
    assert_eq!(ends.get("scene"), Some(&FieldEnd::model(" kitchen ")));
    assert_eq!(ends.len(), 2);
  }

  /// LAW (Codex R1, [medium]): **a member serde_json cannot materialize costs
  /// no other member its account.** A number outside the `f64` range
  /// (`1e400`) and a value nested past serde_json's recursion limit are
  /// JSON the matcher can write, but a whole-answer `Value` refuses them;
  /// each string member's account is bound to its own lexeme all the same.
  #[test]
  fn a_member_a_document_parse_refuses_costs_no_account() {
    let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
    for raw in [
      r#"{"description":"A cat.   ","n":1e400}"#.to_owned(),
      format!(r#"{{"n":{deep},"description":"A cat.   "}}"#),
    ] {
      assert!(
        serde_json::from_str::<Value>(&raw).is_err(),
        "{raw}: a document parse refuses it"
      );
      let ends = field_ends(
        &raw,
        &recorded(&raw, &[ClosedBy::Grammar]),
        &declared(&[("description", 9)]),
      );
      assert_eq!(
        ends.get("description"),
        Some(&FieldEnd::cap("A cat.   ")),
        "{raw}"
      );
    }
  }

  /// LAW (Codex R1, [medium]): **a key written twice binds by its last
  /// occurrence**, as serde_json reads it: the last string binds, and a last
  /// value that is not a string leaves the key with no account.
  #[test]
  fn a_key_written_twice_binds_by_its_last_occurrence() {
    let ends_of = |raw: &str| {
      let mut members = follow(&[raw.as_bytes()]);
      for string in members
        .iter_mut()
        .filter_map(|member| member.string.as_mut())
      {
        string.closed_by = ClosedBy::Model;
      }
      field_ends(raw, &members, &FieldCaps::new())
    };
    assert_eq!(
      ends_of(r#"{"description":"first","description":"last"}"#).get("description"),
      Some(&FieldEnd::model("last"))
    );
    assert_eq!(
      ends_of(r#"{"description":5,"description":"last"}"#).get("description"),
      Some(&FieldEnd::model("last"))
    );
    assert!(ends_of(r#"{"description":"first","description":5}"#).is_empty());
    assert!(ends_of(r#"{"description":"first","description":{"a":"b"}}"#).is_empty());
  }

  /// LAW: **an account binds only a lexeme the answer carries where the
  /// matcher committed it, and only one that decodes**: another text at that
  /// place, the same text shifted, a text cut short, and a lone surrogate
  /// escape (JSON's grammar admits it; no string holds it) all bind nothing.
  #[test]
  fn an_account_binds_only_the_committed_decodable_lexeme() {
    let committed = r#"{"description":"A cat."}"#;
    let members = recorded(committed, &[ClosedBy::Model]);
    assert_eq!(
      field_ends(committed, &members, &FieldCaps::new()).get("description"),
      Some(&FieldEnd::model("A cat."))
    );
    for other in [
      r#"{"description":"A dog."}"#,
      r#" {"description":"A cat."}"#,
      r#"{"description":"A cat"#,
    ] {
      assert!(
        field_ends(other, &members, &FieldCaps::new()).is_empty(),
        "{other}"
      );
    }
    let lone = r#"{"description":"\ud800"}"#;
    assert!(field_ends(lone, &recorded(lone, &[ClosedBy::Model]), &FieldCaps::new()).is_empty());
  }

  /// LAW (R1): **the length at the close decides first, then the mask.** A
  /// string holding exactly its declared `maxLength` is `cap` whoever the
  /// mask shows closed it; below the cap the model's close is `model`, and a
  /// close every allowed token made — an `enum` or a `pattern` can force one
  /// — or an unproven one has no account; nor has a field with no declared
  /// cap the grammar closed, a value that is not a string, or an answer that
  /// is no object.
  #[test]
  fn the_length_at_the_close_decides_first_then_the_mask() {
    let end_of = |raw: &str, by: ClosedBy| {
      field_ends(raw, &recorded(raw, &[by]), &declared(&[("description", 5)]))
        .get("description")
        .cloned()
    };
    for by in [ClosedBy::Grammar, ClosedBy::Unproven, ClosedBy::Model] {
      assert_eq!(
        end_of(r#"{"description":"abcde"}"#, by),
        Some(FieldEnd::cap("abcde")),
        "at the cap under {by:?}"
      );
    }
    let below = r#"{"description":"abcd"}"#;
    assert_eq!(
      end_of(below, ClosedBy::Model),
      Some(FieldEnd::model("abcd"))
    );
    assert_eq!(end_of(below, ClosedBy::Grammar), None);
    assert_eq!(end_of(below, ClosedBy::Unproven), None);
    for raw in [
      r#"{"scene":"kitchen"}"#,
      r#"{"description":["abcde"]}"#,
      r#"["abcde"]"#,
      r#""abcde""#,
    ] {
      assert!(
        field_ends(
          raw,
          &follow(&[raw.as_bytes()]),
          &declared(&[("description", 5)])
        )
        .is_empty(),
        "{raw}"
      );
    }
  }

  // ===== the real matcher over scripted tokens =====

  /// The detokenizer `generate` decodes the answer with.
  fn detokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(super::testing::bundled_tokenizer_json())
      .expect("the tokenizer loads")
  }

  /// Plays `steps` through a tracking `ConstrainedSampler` over `schema`
  /// until the matcher completes the answer; returns the raw answer as
  /// `generate` detokenizes it, and the members the sampler recorded.
  fn decode(schema: &Value, steps: Vec<Step>) -> (String, Vec<Member>) {
    let vocab = trie().vocab_size();
    let mut sampler = ConstrainedSampler::new(
      constraint(schema),
      RequestOptions::deterministic(),
      0,
      vocab as u32,
    )
    .with_field_tracking();
    let mut script = Script::new(steps);
    let mut drawn = Vec::new();
    for step in 0..512 {
      let mut logits = script.logits(vocab);
      let (token, complete) = match sampler.sample(&mut logits, &HashSet::new(), step) {
        Ok(SampleResult::Token(token)) => (token, false),
        Ok(SampleResult::TokenAndComplete(token)) => (token, true),
        Ok(SampleResult::SchemaComplete) => break,
        Err(e) => panic!("step {step}: {e}"),
      };
      script.drew(token);
      drawn.push(token);
      if complete {
        break;
      }
    }
    assert!(script.finished(), "the matcher completed the answer early");
    let raw = detokenizer()
      .decode(&drawn, true)
      .expect("the answer detokenizes");
    (raw, sampler.field_members().to_vec())
  }

  /// The decoded answer's `description`, as serde_json reads it.
  fn description(raw: &str) -> String {
    let answer: Value = serde_json::from_str(raw).expect("the answer is JSON");
    answer["description"]
      .as_str()
      .expect("the description is a string")
      .to_owned()
  }

  /// LAW: **a description that reaches its `maxLength` is closed by the
  /// grammar, and its account is `cap`** — bound to the description exactly
  /// as serde_json decodes it from the answer, its leading space kept — so
  /// the task marks it `Ragged`.
  #[test]
  fn a_description_the_grammar_closes_at_its_max_length_is_capped() {
    let task = ImageAnalysisTask::new();
    let written = caption(task.description_max_chars().get());
    let (raw, members) = decode(
      task.schema(),
      vec![
        text(r#"{"description":""#),
        text(&written),
        text(r#"","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes(&members), [("description", ClosedBy::Grammar)]);
    assert_eq!(description(&raw), written);
    let ends = field_ends(&raw, &members, &task.field_caps());
    let end = ends
      .get("description")
      .expect("the description has an account");
    assert!(end.closed_at_cap());
    assert_eq!(end.source(), description(&raw));
    assert_eq!(end.cut(), None, "a capped string never ends in a cut token");
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Ragged);
    assert_eq!(analysis.description(), written.trim());
  }

  /// LAW (Codex R1, [medium]): **through the matcher, a number serde_json
  /// cannot hold costs the description nothing.** The schema admits any JSON
  /// number, so the matcher writes `1e400`; a whole-answer `Value` refuses
  /// it, and the description the grammar closed at its cap still has its
  /// `cap` account.
  #[test]
  fn a_number_past_f64_costs_the_description_nothing() {
    let schema = json!({
      "type": "object",
      "properties": {
        "description": {"type": "string", "maxLength": 120},
        "n": {"type": "number"},
      },
      "required": ["description", "n"],
      "additionalProperties": false,
    });
    let written = caption(120);
    let (raw, members) = decode(
      &schema,
      vec![
        text(r#"{"description":""#),
        text(&written),
        text(r#"","n":1e400}"#),
      ],
    );
    assert!(serde_json::from_str::<Value>(&raw).is_err(), "{raw}");
    assert_eq!(closes(&members), [("description", ClosedBy::Grammar)]);
    let ends = field_ends(&raw, &members, &declared(&[("description", 120)]));
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap(&written)));
  }

  /// LAW: **a description the model closes short of the cap has the model's
  /// account**, bound untrimmed, and the task marks it `Whole`.
  #[test]
  fn a_description_the_model_closes_is_the_models() {
    let task = ImageAnalysisTask::new();
    let written = " A grey cat sleeps on the rug. ";
    let (raw, members) = decode(
      task.schema(),
      vec![
        text(r#"{"description":""#),
        text(written),
        text(r#"","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes(&members), [("description", ClosedBy::Model)]);
    let ends = field_ends(&raw, &members, &task.field_caps());
    assert_eq!(ends.get("description"), Some(&FieldEnd::model(written)));
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Whole);
  }

  /// LAW (R1, the length decides first): **a close drawn with content at the
  /// cap is the cap's.** One character short of it, the model draws `."` —
  /// the last character and the close in one token. The mask cannot say who
  /// closed it: it also allows `.` alone, which stops inside the string, and
  /// no token that writes past the `.`. But the description closed holding
  /// exactly its 120 characters: the model could not have written past them
  /// whichever token carried the quote, so it is `cap`, and `Ragged`.
  #[test]
  fn a_close_drawn_with_content_at_the_cap_is_the_caps() {
    let task = ImageAnalysisTask::new();
    let cap = task.description_max_chars().get();
    let period_quote = trie()
      .token_id(br#".""#)
      .expect("the vocabulary holds `.\"`");
    let written = format!("{}.", caption(cap - 1));
    let (raw, members) = decode(
      task.schema(),
      vec![
        text(r#"{"description":""#),
        text(&caption(cap - 1)),
        Step::Token(period_quote),
        text(r#","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes(&members), [("description", ClosedBy::Unproven)]);
    assert_eq!(description(&raw), written);
    let ends = field_ends(&raw, &members, &task.field_caps());
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap(&written)));
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Ragged);
  }

  /// A schema with a `description` of at most `cap` characters and `tags`.
  fn capped(cap: usize) -> Value {
    json!({
      "type": "object",
      "properties": {
        "description": {"type": "string", "maxLength": cap},
        "tags": {"type": "array", "items": {"type": "string"}},
      },
      "required": ["description", "tags"],
      "additionalProperties": false,
    })
  }

  /// LAW (Codex R1, [medium]): **a `maxLength` of 0 is the grammar's close:
  /// `cap("")`.** The matcher forces the empty string, narrowing each mask to
  /// the token that starts the forced bytes' canonical tokenization; the
  /// token that opens and closes the description was the only one allowed.
  #[test]
  fn a_max_length_of_zero_is_closed_by_the_grammar() {
    let schema = capped(0);
    let (raw, members) = decode(&schema, vec![text(r#"{"description":"","tags":["cat"]}"#)]);
    assert_eq!(closes(&members), [("description", ClosedBy::Grammar)]);
    let ends = field_ends(&raw, &members, &declared(&[("description", 0)]));
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap("")));
  }

  /// The account a task declaring `caps` gets for `field` of `answer` under
  /// `schema`, decoded through the real matcher: `answer` is `open`, the
  /// field's `content`, then `close` — the closing quote its own step, so the
  /// field's last content token never carries it.
  fn decoded_end(
    schema: &Value,
    (open, content, close): (&str, &str, &str),
    field: &str,
    caps: &FieldCaps,
  ) -> Option<FieldEnd> {
    let (raw, members) = decode(schema, vec![text(open), text(content), text(close)]);
    assert_eq!(raw, format!("{open}{content}{close}"));
    field_ends(&raw, &members, caps).get(field).cloned()
  }

  /// LAW (Codex R3): **a field the schema caps is the cap's only when the task
  /// declares its cap** — lfm reads no cap out of the schema. For each schema
  /// the field reaches its cap through — a `$ref` to `#/$defs/…`, an `allOf`
  /// of 8 and 5, `additionalProperties` for a field `properties` does not name
  /// — the matcher closes the field at the cap; under the task's declared cap
  /// the account is `cap`, and with none declared there is no account. A field
  /// the model closed short of the cap is `model` either way.
  #[test]
  fn a_schema_capped_field_is_the_caps_only_when_the_task_declares_it() {
    let by_ref = json!({
      "type": "object",
      "properties": {"description": {"$ref": "#/$defs/description"}},
      "required": ["description"],
      "additionalProperties": false,
      "$defs": {"description": {"type": "string", "maxLength": 5}},
    });
    let all_of = json!({
      "type": "object",
      "properties": {"description": {"allOf": [
        {"type": "string", "maxLength": 8},
        {"maxLength": 5},
      ]}},
      "required": ["description"],
      "additionalProperties": false,
    });
    let additional = json!({
      "type": "object",
      "additionalProperties": {"type": "string", "maxLength": 3},
    });
    for (schema, field, at_cap, short) in [
      (&by_ref, "description", "abcde", "ab"),
      (&all_of, "description", "abcde", "ab"),
      (&additional, "note", "abc", "a"),
    ] {
      let open = format!(r#"{{"{field}":""#);
      let cap = at_cap.chars().count();
      let declared = declared(&[(field, cap)]);
      assert_eq!(
        decoded_end(schema, (&open, at_cap, r#""}"#), field, &declared),
        Some(FieldEnd::cap(at_cap)),
        "{schema}: declared"
      );
      assert_eq!(
        decoded_end(schema, (&open, at_cap, r#""}"#), field, &FieldCaps::new()),
        None,
        "{schema}: undeclared"
      );
      for caps in [&declared, &FieldCaps::new()] {
        assert_eq!(
          decoded_end(schema, (&open, short, r#""}"#), field, caps),
          Some(FieldEnd::model(short)),
          "{schema}: the model's close"
        );
      }
    }
  }

  /// LAW (Codex R3, [high]): **llguidance resolves a `$ref` inside an `$id`
  /// subresource against that subresource, and the account follows the
  /// task's declared cap.** The description refers to `#/$defs/inner`, which
  /// sets its own `$id`; inside it `#/$defs/cap` is the subresource's cap of
  /// 3, not the root's 9. The matcher closes `"abc"` itself — every token it
  /// allows at the third character closes the string — and under the task's
  /// declared 3 the account is `cap("abc")`.
  #[test]
  fn an_id_subresource_cap_is_the_matchers_and_the_declared_one() {
    let schema = json!({
      "type": "object",
      "properties": {"description": {"$ref": "#/$defs/inner"}},
      "required": ["description"],
      "additionalProperties": false,
      "$defs": {
        "cap": {"type": "string", "maxLength": 9},
        "inner": {
          "$id": "https://example.com/fieldend/inner",
          "$ref": "#/$defs/cap",
          "$defs": {"cap": {"type": "string", "maxLength": 3}},
        },
      },
    });
    let (raw, members) = decode(
      &schema,
      vec![text(r#"{"description":""#), text("abc"), text(r#""}"#)],
    );
    assert_eq!(closes(&members), [("description", ClosedBy::Grammar)]);
    let ends = field_ends(&raw, &members, &declared(&[("description", 3)]));
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap("abc")));
  }

  /// LAW (Codex R1, [medium]): **a string the model opens and closes in one
  /// token under a larger cap is the model's.** The bundled vocabulary's
  /// one-token string is `""`: drawn as the description under a cap of 120,
  /// it shares its mask with tokens that open the string and write on (`"A`
  /// and its like), so the account is `model("")`.
  #[test]
  fn a_one_token_string_the_model_chose_is_the_models() {
    let schema = capped(120);
    let empty = trie()
      .token_id(br#""""#)
      .expect("the vocabulary holds `\"\"`");
    let (raw, members) = decode(
      &schema,
      vec![
        text(r#"{"description":"#),
        Step::Token(empty),
        text(r#","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes(&members), [("description", ClosedBy::Model)]);
    let ends = field_ends(&raw, &members, &declared(&[("description", 120)]));
    assert_eq!(ends.get("description"), Some(&FieldEnd::model("")));
  }

  /// LAW (Codex R1, [medium]): **a one-token string the mask cannot
  /// attribute has no account, not the model's by default.** The token
  /// `","` opens the description, writes `,` and closes it. Its mask allows
  /// the lone quote, which opens the string and stops, so not every allowed
  /// token closes it; and no allowed token writes past the `,` (the
  /// vocabulary's only one, `",\n`, puts a raw newline in a string). The
  /// description gets no account.
  #[test]
  fn a_one_token_string_the_mask_cannot_attribute_has_no_account() {
    let schema = capped(120);
    let comma = trie()
      .token_id(br#"",""#)
      .expect("the vocabulary holds `\",\"`");
    let (raw, members) = decode(
      &schema,
      vec![
        text(r#"{"description":"#),
        Step::Token(comma),
        text(r#","tags":["cat"]}"#),
      ],
    );
    assert_eq!(description(&raw), ",");
    assert_eq!(closes(&members), [("description", ClosedBy::Unproven)]);
    assert!(field_ends(&raw, &members, &declared(&[("description", 120)])).is_empty());
  }

  /// The mask `matcher` computes for its next token.
  fn next_mask(matcher: &mut llguidance::Constraint) -> SimpleVob {
    matcher.compute_mask().expect("a mask");
    matcher
      .step_result()
      .sample_mask
      .clone()
      .expect("a sampling mask")
  }

  /// The first bytes of the tokens `mask` allows.
  fn first_bytes(mask: &SimpleVob) -> HashSet<u8> {
    mask
      .iter()
      .filter_map(|allowed| trie().token(allowed).first().copied())
      .collect()
  }

  /// LAW: **a partial character never meets the close, so no account names
  /// a cut token.** One character short of the cap, the three bytes of `語`
  /// arrive as three single-byte tokens. While the character is incomplete
  /// the matcher allows only tokens that open with a continuation byte —
  /// never the close — and once it is complete every token it allows opens
  /// with the close. Through the sampler the account is `cap` with no cut,
  /// and the description ends in `語`, not in U+FFFD.
  #[test]
  fn a_partial_character_never_meets_the_close() {
    let task = ImageAnalysisTask::new();
    let written = caption(task.description_max_chars().get() - 1);
    let character = "語".as_bytes();
    assert_eq!(character, [0xE8, 0xAA, 0x9E]);
    let byte_tokens: Vec<u32> = character
      .iter()
      .map(|&byte| {
        trie()
          .token_id(&[byte])
          .unwrap_or_else(|| panic!("the byte-level vocabulary holds {byte:#04x}"))
      })
      .collect();

    // The matcher itself, its mask read at every step.
    let vocab = trie().vocab_size();
    let mut matcher = constraint(task.schema());
    let mut tracker = FieldTracker::new();
    let mut prefix = Script::new(vec![text(r#"{"description":""#), text(&written)]);
    while !prefix.finished() {
      let mask = next_mask(&mut matcher);
      let logits = prefix.logits(vocab);
      let token = mask
        .iter()
        .max_by(|a, b| logits[*a as usize].total_cmp(&logits[*b as usize]))
        .expect("an allowed token");
      prefix.drew(token);
      tracker.commit(
        trie().token(token),
        &Masked {
          mask: &mask,
          trie: trie(),
        },
      );
      matcher
        .commit_token(Some(token))
        .expect("the token commits");
    }
    for (at, &token) in byte_tokens.iter().enumerate() {
      let mask = next_mask(&mut matcher);
      assert!(
        mask.is_allowed(token),
        "byte {at} of the character is allowed"
      );
      if at > 0 {
        let first = first_bytes(&mask);
        assert!(
          first.iter().all(|byte| (0x80..=0xBF).contains(byte)),
          "after {at} of the character's bytes only continuation bytes follow: {first:x?}"
        );
      }
      tracker.commit(
        trie().token(token),
        &Masked {
          mask: &mask,
          trie: trie(),
        },
      );
      matcher.commit_token(Some(token)).expect("the byte commits");
    }
    let mask = next_mask(&mut matcher);
    let quote_byte = b'"';
    assert_eq!(
      first_bytes(&mask),
      HashSet::from([quote_byte]),
      "at the cap every allowed token opens with the close"
    );
    let quote = trie().token_id(br#"""#).expect("the vocabulary holds `\"`");
    tracker.commit(
      trie().token(quote),
      &Masked {
        mask: &mask,
        trie: trie(),
      },
    );
    assert_eq!(
      closes(tracker.members()),
      [("description", ClosedBy::Grammar)]
    );

    let mut steps = vec![text(r#"{"description":""#), text(&written)];
    steps.extend(byte_tokens.iter().map(|&token| Step::Token(token)));
    steps.push(text(r#"","tags":["cat"]}"#));
    let (raw, members) = decode(task.schema(), steps);
    assert_eq!(closes(&members), [("description", ClosedBy::Grammar)]);
    let ends = field_ends(&raw, &members, &task.field_caps());
    let end = ends
      .get("description")
      .expect("the description has an account");
    assert!(end.closed_at_cap());
    assert_eq!(end.cut(), None);
    assert!(end.source().ends_with('語'));
    assert!(!end.source().contains('\u{FFFD}'));
    assert_eq!(end.source(), description(&raw));
  }

  /// LAW: **every string field of the answer gets an account and nothing else
  /// does**: the extensions' string fields closed by the model are `model`,
  /// the capped description `cap`; an array of strings (`subjects`, `tags`)
  /// is no string field.
  #[test]
  fn every_string_field_gets_an_account_and_nothing_else_does() {
    let task = ImageAnalysisTask::new().with_extensions([
      Extension::Scene,
      Extension::Subjects,
      Extension::ShotType,
    ]);
    let written = caption(task.description_max_chars().get());
    let mut steps = vec![text("{")];
    let properties = task.schema()["properties"]
      .as_object()
      .expect("the schema's properties are an object");
    for (at, field) in properties.keys().enumerate() {
      if at > 0 {
        steps.push(text(","));
      }
      steps.push(text(&format!(r#""{field}":"#)));
      match field.as_str() {
        "description" => steps.extend([text(r#"""#), text(&written), text(r#"""#)]),
        "scene" => steps.push(text(r#""kitchen""#)),
        "shot_type" => steps.push(text(r#""close-up""#)),
        "subjects" | "tags" => steps.push(text(r#"["cat"]"#)),
        other => panic!("unexpected field {other}"),
      }
    }
    steps.push(text("}"));
    let (raw, members) = decode(task.schema(), steps);
    let ends = field_ends(&raw, &members, &task.field_caps());
    let fields: Vec<(&str, bool)> = ends
      .iter()
      .map(|(field, end)| (field, end.closed_at_cap()))
      .collect();
    assert_eq!(
      fields,
      [
        ("description", true),
        ("scene", false),
        ("shot_type", false)
      ]
    );
    assert_eq!(ends.get("scene"), Some(&FieldEnd::model("kitchen")));
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Ragged);
  }
}
