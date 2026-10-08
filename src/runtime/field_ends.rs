//! The decoder's account of how it ended each string field of a JSON answer
//! (findit-studio/application#235): whether the model closed the string or
//! the grammar closed it at the field's `maxLength`.
//!
//! [`ConstrainedSampler`](super::sampler::ConstrainedSampler) hands every
//! token it commits to a [`FieldTracker`], which follows the JSON structure of
//! the committed bytes far enough to know which member of the answer's
//! top-level object a string value belongs to, and which token closes it. For
//! that token the sampler still holds the mask the token was drawn under, so
//! it asks the tracker whether the grammar left the model any other choice:
//! the close is the grammar's when no token the mask allowed would have left
//! the string open ([`FieldTracker::grammar_closes`]). [`field_ends`] then
//! binds each record to the member's string as the finished answer carries
//! it, as llmtask's [`FieldEnd`].
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
//! The mask is read the same way: whether an allowed token would have left
//! the string open is decided by that token's bytes from the string's current
//! state. When llguidance forces bytes it can narrow the mask to the single
//! token that starts their canonical tokenization (`TokenParser::
//! compute_mask`), and that token may carry content before the quote; every
//! token the mask allows then closes the string, so the close still reads as
//! the grammar's.
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

use llguidance::toktrie::{SimpleVob, TokTrie};
use llmtask::{FieldEnd, FieldEnds};
use serde_json::Value;
use smol_str::SmolStr;

/// How the decode closed one string member of the answer's top-level object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldClose {
  /// The member's key, JSON-decoded.
  pub(crate) field: SmolStr,
  /// The grammar closed the string: no token the mask allowed at the closing
  /// step would have left it open.
  pub(crate) forced: bool,
}

/// Follows the JSON structure of the committed bytes far enough to record how
/// each string member of the top-level object closed.
#[derive(Debug, Default)]
pub(crate) struct FieldTracker {
  /// The containers open around the next byte, outermost first.
  open: Vec<Container>,
  /// The string the next byte is inside, if any.
  string: Option<OpenString>,
  /// The decoded key of the top-level member whose value is being read.
  member: Option<SmolStr>,
  /// The top-level value has closed: no later byte belongs to it.
  done: bool,
  closes: Vec<FieldClose>,
}

#[derive(Debug, Clone, Copy)]
enum Container {
  /// An object; `key_next` while the next string in it is a key.
  Object {
    key_next: bool,
  },
  Array,
}

#[derive(Debug)]
struct OpenString {
  role: Role,
  /// The previous byte was an unescaped backslash, so this one is escaped.
  escaped: bool,
}

#[derive(Debug)]
enum Role {
  /// A key of the top-level object, its bytes as written (escapes included).
  Key(Vec<u8>),
  /// The string value of the top-level member being read.
  Member,
  /// Any other string: a nested key or value, an array's element.
  Other,
}

impl FieldTracker {
  /// Nothing committed yet.
  // Its one caller, `Engine::run`, is compiled only with `decoders` on.
  #[cfg_attr(not(feature = "decoders"), allow(dead_code))]
  pub(crate) fn new() -> Self {
    Self::default()
  }

  /// How each top-level string member closed, in the order they closed.
  #[cfg_attr(not(feature = "decoders"), allow(dead_code))]
  pub(crate) fn closes(&self) -> &[FieldClose] {
    &self.closes
  }

  /// Whether the grammar closes the open top-level member's string at this
  /// step: `token`, the token drawn under `mask`, closes it, and no token
  /// `mask` allowed would have left it open.
  pub(crate) fn grammar_closes(&self, token: &[u8], mask: &SimpleVob, trie: &TokTrie) -> bool {
    let Some(OpenString {
      role: Role::Member,
      escaped,
    }) = self.string
    else {
      return false;
    };
    closing_quote(escaped, token).is_some()
      && !mask
        .iter()
        .any(|allowed| leaves_open(escaped, trie.token(allowed)))
  }

  /// Follows one committed token's `bytes`. `forced` is the account of the
  /// first top-level member string they close: whether the grammar closed it
  /// ([`Self::grammar_closes`], asked before the commit). A later member
  /// string opened and closed within the same token is the model's: the mask
  /// the token was drawn under says nothing about it.
  pub(crate) fn commit(&mut self, bytes: &[u8], forced: bool) {
    let mut forced = forced;
    for &byte in bytes {
      self.byte(byte, &mut forced);
    }
  }

  fn byte(&mut self, byte: u8, forced: &mut bool) {
    if self.done {
      return;
    }
    if let Some(string) = &mut self.string {
      if !string.escaped && byte == b'"' {
        self.close_string(forced);
        return;
      }
      string.escaped = !string.escaped && byte == b'\\';
      if let Role::Key(written) = &mut string.role {
        written.push(byte);
      }
      return;
    }
    match byte {
      b'{' => self.open.push(Container::Object { key_next: true }),
      b'[' => self.open.push(Container::Array),
      b'}' | b']' => {
        self.open.pop();
        self.done = self.open.is_empty();
      }
      b',' => {
        if let Some(Container::Object { key_next }) = self.open.last_mut() {
          *key_next = true;
        }
        if self.open.len() == 1 {
          self.member = None;
        }
      }
      b'"' => self.open_string(),
      // `:`, whitespace, and the bytes of a number, `true`, `false` or `null`.
      _ => {}
    }
  }

  fn open_string(&mut self) {
    let top_level = self.open.len() == 1;
    let role = match self.open.last_mut() {
      Some(Container::Object { key_next }) if *key_next => {
        *key_next = false;
        if top_level {
          Role::Key(Vec::new())
        } else {
          Role::Other
        }
      }
      Some(Container::Object { .. }) if top_level => Role::Member,
      _ => Role::Other,
    };
    self.string = Some(OpenString {
      role,
      escaped: false,
    });
  }

  fn close_string(&mut self, forced: &mut bool) {
    let Some(string) = self.string.take() else {
      return;
    };
    match string.role {
      Role::Key(written) => self.member = decode_key(&written),
      Role::Member => {
        if let Some(field) = self.member.take() {
          self.closes.push(FieldClose {
            field,
            forced: *forced,
          });
        }
        *forced = false;
      }
      Role::Other => {}
    }
  }
}

/// Where in `bytes` an open string closes, reading on from a state whose
/// previous byte was an unescaped backslash when `escaped`: the offset of the
/// first unescaped quote, if any.
fn closing_quote(escaped: bool, bytes: &[u8]) -> Option<usize> {
  let mut escaped = escaped;
  for (at, &byte) in bytes.iter().enumerate() {
    if !escaped && byte == b'"' {
      return Some(at);
    }
    escaped = !escaped && byte == b'\\';
  }
  None
}

/// Whether a token of these `bytes` would leave an open string open: string
/// content with no closing quote. A special token (its bytes start with
/// [`TokTrie::SPECIAL_TOKEN_MARKER`]) or one with no bytes continues nothing.
fn leaves_open(escaped: bool, bytes: &[u8]) -> bool {
  bytes
    .first()
    .is_some_and(|&first| first != TokTrie::SPECIAL_TOKEN_MARKER)
    && closing_quote(escaped, bytes).is_none()
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
/// `closes` records, for the task's `parse_ended`.
///
/// A member the model closed is [`FieldEnd::model`]; one the grammar closed
/// at the `maxLength` its entry in `schema`'s top-level `properties` declares
/// — the string holding exactly that many characters — is [`FieldEnd::cap`].
/// Each account holds the member's string as `raw` carries it, JSON-decoded
/// by serde_json and untrimmed: the string the task's own parse reads from
/// the same text.
///
/// A member gets no account when the grammar closed it short of a declared
/// cap (an `enum`, a `const` or a `pattern` can leave the model no other
/// choice too, and that close is not at a cap), when it closed more than once
/// (an account describes one string), or when its value in `raw` is not a
/// string; and `raw` gets none at all when serde_json does not read it as an
/// object.
// Its one caller, `Engine::run`, is compiled only with `decoders` on.
#[cfg_attr(not(feature = "decoders"), allow(dead_code))]
pub(crate) fn field_ends(raw: &str, closes: &[FieldClose], schema: &Value) -> FieldEnds {
  let mut ends = FieldEnds::new();
  let Ok(Value::Object(members)) = serde_json::from_str::<Value>(raw) else {
    return ends;
  };
  for close in closes {
    if closes
      .iter()
      .filter(|other| other.field == close.field)
      .count()
      > 1
    {
      continue;
    }
    let field = close.field.as_str();
    let Some(Value::String(text)) = members.get(field) else {
      continue;
    };
    let end = if !close.forced {
      FieldEnd::model(text)
    } else if max_length(schema, field) == Some(text.chars().count()) {
      FieldEnd::cap(text)
    } else {
      continue;
    };
    ends.insert(close.field.clone(), end);
  }
  ends
}

/// The `maxLength` `schema` declares for its top-level property `field`.
fn max_length(schema: &Value, field: &str) -> Option<usize> {
  schema
    .get("properties")?
    .get(field)?
    .get("maxLength")?
    .as_u64()
    .and_then(|cap| usize::try_from(cap).ok())
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
  use serde_json::json;

  use super::{
    testing::{Script, Step, caption, constraint, text, trie},
    *,
  };
  use crate::{
    options::RequestOptions,
    runtime::sampler::{ConstrainedSampler, SampleResult, Sampler},
  };

  // ===== the tracker, byte by byte =====

  /// The closes a tracker records over `tokens`, each a committed token's
  /// bytes, every first member close in a token accounted `forced`.
  fn follow(tokens: &[&[u8]], forced: bool) -> Vec<FieldClose> {
    let mut tracker = FieldTracker::new();
    for token in tokens {
      tracker.commit(token, forced);
    }
    tracker.closes().to_vec()
  }

  fn close(field: &str, forced: bool) -> FieldClose {
    FieldClose {
      field: field.into(),
      forced,
    }
  }

  /// LAW: **every string member of the top-level object closes once, in
  /// order, and nothing else does** — not an array's elements, not a nested
  /// object's keys or values, not a key — however the bytes fall into tokens
  /// and whatever whitespace sits between them.
  #[test]
  fn each_top_level_string_member_closes_once_in_order() {
    let answer: &[u8] = br#"{ "scene" : "kitchen", "tags":["cat","rug"], "meta":{"note":"x","list":["y"]}, "description":"A cat.", "n": 1.5e3, "ok": true }"#;
    let expected = [close("scene", false), close("description", false)];
    assert_eq!(follow(&[answer], false), expected, "one token");
    for size in 1..9 {
      let tokens: Vec<&[u8]> = answer.chunks(size).collect();
      assert_eq!(follow(&tokens, false), expected, "tokens of {size} bytes");
    }
  }

  /// LAW: **a token's account is its first member close's**: a later member
  /// string opened and closed within the same token is the model's.
  #[test]
  fn a_tokens_account_is_its_first_member_closes() {
    let tokens: [&[u8]; 3] = [br#"{"description":"A cat"#, br#"","scene":"x""#, b"}"];
    assert_eq!(
      follow(&tokens, true),
      [close("description", true), close("scene", false)]
    );
  }

  /// LAW: **an escaped quote or backslash never closes a string**, wherever
  /// the token boundary falls inside the escape; and a key is named as JSON
  /// decodes it.
  #[test]
  fn escapes_never_close_a_string_and_keys_are_decoded() {
    let answer: &[u8] = br#"{"d\u0065scription":"say \"hi\" \\","sc\"ene":"\\\""}"#;
    let expected = [close("description", false), close("sc\"ene", false)];
    for size in 1..12 {
      let tokens: Vec<&[u8]> = answer.chunks(size).collect();
      assert_eq!(follow(&tokens, false), expected, "tokens of {size} bytes");
    }
    assert_eq!(closing_quote(false, br#"ab"c"#), Some(2));
    assert_eq!(closing_quote(true, br#""c"#), None, "an escaped quote");
    assert_eq!(
      closing_quote(false, br#"\\""#),
      Some(2),
      "after an escaped backslash"
    );
    assert!(leaves_open(false, b"abc"));
    assert!(leaves_open(true, br#""abc"#), "the quote is escaped");
    assert!(!leaves_open(false, br#"c""#));
    assert!(!leaves_open(false, b""), "no bytes continue nothing");
    assert!(
      !leaves_open(false, &[TokTrie::SPECIAL_TOKEN_MARKER, b'a']),
      "a special token continues nothing"
    );
  }

  /// LAW: **only a member of a top-level object is a field**: the strings of
  /// a top-level array or a top-level string are no field's, and nothing
  /// after the top-level value counts.
  #[test]
  fn only_a_member_of_a_top_level_object_is_a_field() {
    assert!(follow(&[br#"["a","b"]"#], true).is_empty());
    assert!(follow(&[br#""abc""#], true).is_empty());
    let tokens: [&[u8]; 3] = [br#"{"a":"x"}"#, b" ", br#""b":"y""#];
    assert_eq!(follow(&tokens, false), [close("a", false)]);
  }

  // ===== binding the closes to the answer =====

  /// LAW: **an account holds the field exactly as serde_json decodes it from
  /// the answer** — escapes decoded, untrimmed — the model's close as
  /// `model`, the grammar's at the declared cap as `cap`.
  #[test]
  fn an_account_holds_the_decoded_untrimmed_field() {
    let schema = json!({"properties": {"description": {"type": "string", "maxLength": 9}}});
    let raw = r#"{"description":"  A \"cat\"","scene":" kitchen "}"#;
    let decoded: Value = serde_json::from_str(raw).expect("the answer is JSON");
    assert_eq!(decoded["description"], "  A \"cat\"");
    let ends = field_ends(
      raw,
      &[close("description", true), close("scene", false)],
      &schema,
    );
    assert_eq!(ends.get("description"), Some(&FieldEnd::cap("  A \"cat\"")));
    assert_eq!(ends.get("scene"), Some(&FieldEnd::model(" kitchen ")));
    assert_eq!(ends.len(), 2);
  }

  /// LAW: **the grammar's close is a cap's account only at the declared
  /// cap.** Short of the `maxLength`, or for a field with none (an `enum` or
  /// a `pattern` can force a close too), it is no account; a field closed
  /// twice is no account, nor one whose value is not a string; and an answer
  /// serde_json does not read as an object has none.
  #[test]
  fn a_forced_close_is_a_caps_account_only_at_the_declared_cap() {
    let schema = json!({"properties": {
      "description": {"type": "string", "maxLength": 5},
      "scene": {"type": "string", "enum": ["kitchen"]},
    }});
    let at_cap = field_ends(
      r#"{"description":"abcde"}"#,
      &[close("description", true)],
      &schema,
    );
    assert_eq!(at_cap.get("description"), Some(&FieldEnd::cap("abcde")));
    for (raw, closes) in [
      (
        r#"{"description":"abcd"}"#,
        vec![close("description", true)],
      ),
      (r#"{"scene":"kitchen"}"#, vec![close("scene", true)]),
      (
        r#"{"description":"abcde","description":"abcde"}"#,
        vec![close("description", true), close("description", true)],
      ),
      (
        r#"{"description":["abcde"]}"#,
        vec![close("description", false)],
      ),
      ("not json", vec![close("description", false)]),
      (r#"["abcde"]"#, vec![close("description", false)]),
    ] {
      assert!(
        field_ends(raw, &closes, &schema).is_empty(),
        "{raw} under {closes:?}"
      );
    }
  }

  // ===== the real matcher over scripted tokens =====

  /// The detokenizer `generate` decodes the answer with.
  fn detokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(super::testing::bundled_tokenizer_json())
      .expect("the tokenizer loads")
  }

  /// Plays `steps` through a tracking `ConstrainedSampler` over `task`'s
  /// grammar until the matcher completes the answer; returns the raw answer
  /// as `generate` detokenizes it, and the sampler's closes.
  fn decode(task: &ImageAnalysisTask, steps: Vec<Step>) -> (String, Vec<FieldClose>) {
    let vocab = trie().vocab_size();
    let mut sampler = ConstrainedSampler::new(
      constraint(task.schema()),
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
    (raw, sampler.field_closes().to_vec())
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
    let (raw, closes) = decode(
      &task,
      vec![
        text(r#"{"description":""#),
        text(&written),
        text(r#"","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes, [close("description", true)]);
    assert_eq!(description(&raw), written);
    let ends = field_ends(&raw, &closes, task.schema());
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

  /// LAW: **a description the model closes short of the cap has the model's
  /// account**, bound untrimmed, and the task marks it `Whole`.
  #[test]
  fn a_description_the_model_closes_is_the_models() {
    let task = ImageAnalysisTask::new();
    let written = " A grey cat sleeps on the rug. ";
    let (raw, closes) = decode(
      &task,
      vec![
        text(r#"{"description":""#),
        text(written),
        text(r#"","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes, [close("description", false)]);
    let ends = field_ends(&raw, &closes, task.schema());
    assert_eq!(ends.get("description"), Some(&FieldEnd::model(written)));
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Whole);
  }

  /// LAW: **the model's close is the model's even at the cap.** One
  /// character short of it, the model draws `."` — content, then the close,
  /// in one token — where the matcher also allowed tokens that keep the
  /// string open: the account is `model` although the description holds
  /// exactly the cap's characters.
  #[test]
  fn a_close_drawn_with_content_is_the_models_even_at_the_cap() {
    let task = ImageAnalysisTask::new();
    let cap = task.description_max_chars().get();
    let period_quote = trie()
      .token_id(br#".""#)
      .expect("the vocabulary holds `.\"`");
    let (raw, closes) = decode(
      &task,
      vec![
        text(r#"{"description":""#),
        text(&caption(cap - 1)),
        Step::Token(period_quote),
        text(r#","tags":["cat"]}"#),
      ],
    );
    assert_eq!(closes, [close("description", false)]);
    let ends = field_ends(&raw, &closes, task.schema());
    let end = ends
      .get("description")
      .expect("the description has an account");
    assert!(!end.closed_at_cap());
    assert_eq!(end.source().chars().count(), cap);
    let analysis = task.parse_ended(&raw, &ends).expect("the answer parses");
    assert_eq!(analysis.description_end(), DescriptionEnd::Whole);
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
      tracker.commit(trie().token(token), false);
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
      tracker.commit(trie().token(token), false);
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
    assert!(tracker.grammar_closes(trie().token(quote), &mask, trie()));

    let mut steps = vec![text(r#"{"description":""#), text(&written)];
    steps.extend(byte_tokens.iter().map(|&token| Step::Token(token)));
    steps.push(text(r#"","tags":["cat"]}"#));
    let (raw, closes) = decode(&task, steps);
    assert_eq!(closes, [close("description", true)]);
    let ends = field_ends(&raw, &closes, task.schema());
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
    let (raw, closes) = decode(&task, steps);
    let ends = field_ends(&raw, &closes, task.schema());
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
