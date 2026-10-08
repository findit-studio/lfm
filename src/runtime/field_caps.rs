//! The `maxLength` a JSON Schema puts on each string member of the answer's
//! top-level object — the cap the length-first rule of the decoder's account
//! reads ([`field_ends`](super::field_ends::field_ends)) — resolved the way
//! llguidance normalizes the schema.
//!
//! # How a member's cap resolves
//!
//! A member's value meets every constraint the schema puts on it, and its cap
//! is the smallest `maxLength` among them (llguidance's `Schema::intersect`
//! keeps the smaller `maxLength`):
//!
//! - the member's entry in `properties`, together with every
//!   `patternProperties` entry whose pattern matches its key; or, when neither
//!   names it, `additionalProperties`;
//! - in each of those, the node's own `maxLength`, every branch of an `allOf`,
//!   and the target of a `$ref` — a local JSON Pointer (`#/$defs/…`,
//!   `#/definitions/…`), followed through chains; a reference met again on its
//!   own path adds nothing;
//! - the same for the top-level object itself: its own `$ref` and `allOf`
//!   branches add the object schemas a member also meets.
//!
//! A member no constraint caps has no cap: the model's close is `model`, and
//! the grammar's close is no account.
//!
//! # What is refused
//!
//! An `anyOf` or `oneOf` whose branches that can hold the value disagree on
//! the cap — or only some of which cap it — leaves the cap to the branch the
//! model's string falls under, which the schema alone does not fix; so does a
//! `$ref` that is not a local JSON Pointer, and a `patternProperties` pattern
//! the `regex` crate cannot compile, for the cap of a key that may match it.
//! [`FieldCaps::of`] refuses such a schema by the member's name
//! ([`Error::UnsupportedFieldCap`]) before a token is drawn, rather than
//! leaving the member's account silently out.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use smol_str::SmolStr;

use crate::error::{Error, Result};

/// A cap that depends on which branch of an `anyOf` or `oneOf` the value
/// falls under.
pub(crate) const BRANCH_DEPENDENT: &str =
  "branch-dependent maxLength is not supported for field-end accounts";

/// A `$ref` that is not a local JSON Pointer into the schema.
pub(crate) const UNRESOLVED_REF: &str =
  "a $ref that is not a local JSON Pointer is not supported for field-end accounts";

/// A `patternProperties` pattern the `regex` crate cannot compile.
pub(crate) const UNMATCHABLE_PATTERN: &str = "a patternProperties pattern the regex crate cannot compile is not supported for field-end accounts";

/// The caps a JSON Schema puts on the string members of its top-level
/// object.
#[derive(Debug, Default)]
pub(crate) struct FieldCaps {
  /// Every object schema the answer's top-level object meets: its own, and
  /// those its `$ref`s and `allOf` branches add.
  objects: Vec<ObjectCaps>,
}

impl FieldCaps {
  /// The caps `schema` puts on the string members of its top-level object.
  ///
  /// # Errors
  ///
  /// [`Error::UnsupportedFieldCap`], naming the member, when the schema alone
  /// does not fix a member's cap (see the module docs).
  // Its one caller, `Engine::run`, is compiled only with `decoders` on.
  #[cfg_attr(not(feature = "decoders"), allow(dead_code))]
  pub(crate) fn of(schema: &Value) -> Result<Self> {
    let resolver = Resolver { root: schema };
    Ok(Self {
      objects: resolver.objects(schema, &mut Vec::new())?,
    })
  }

  /// The cap on the string member `field`: the smallest `maxLength` among the
  /// constraints its value meets, if any caps it.
  pub(crate) fn cap(&self, field: &str) -> Option<usize> {
    cap_among(&self.objects, field)
  }
}

/// The cap on `field` among `objects`, every one of which the object meets.
fn cap_among(objects: &[ObjectCaps], field: &str) -> Option<usize> {
  objects.iter().filter_map(|object| object.cap(field)).min()
}

/// One object schema's caps on its members.
#[derive(Debug)]
struct ObjectCaps {
  /// The cap of each member `properties` declares.
  properties: BTreeMap<String, Option<usize>>,
  /// The cap of each `patternProperties` pattern's members.
  patterns: Vec<PatternCap>,
  /// The cap of a member neither `properties` nor a pattern names.
  additional: Option<usize>,
}

#[derive(Debug)]
struct PatternCap {
  pattern: String,
  /// The pattern compiled, matched anywhere in a key as JSON Schema reads it.
  matcher: llmtask::Grammar,
  cap: Option<usize>,
}

impl ObjectCaps {
  fn cap(&self, field: &str) -> Option<usize> {
    let declared = self.properties.get(field);
    let mut matched = declared.is_some();
    let mut caps: Vec<Option<usize>> = declared.copied().into_iter().collect();
    for pattern in &self.patterns {
      if pattern
        .matcher
        .as_regex()
        .is_some_and(|regex| regex.is_match(field))
      {
        matched = true;
        caps.push(pattern.cap);
      }
    }
    if !matched {
      caps.push(self.additional);
    }
    caps.into_iter().flatten().min()
  }

  /// Whether any member this schema names is capped.
  fn caps_anything(&self) -> bool {
    self.properties.values().any(Option::is_some)
      || self.patterns.iter().any(|pattern| pattern.cap.is_some())
      || self.additional.is_some()
  }
}

/// A JSON type a union's branch may rule out.
#[derive(Debug, Clone, Copy)]
enum Kind {
  String,
  Object,
}

impl Kind {
  /// The name JSON Schema's `type` gives it.
  const fn name(self) -> &'static str {
    match self {
      Self::String => "string",
      Self::Object => "object",
    }
  }

  fn holds(self, value: &Value) -> bool {
    match self {
      Self::String => value.is_string(),
      Self::Object => value.is_object(),
    }
  }
}

/// Resolves a schema's caps against its root document.
struct Resolver<'s> {
  root: &'s Value,
}

impl<'s> Resolver<'s> {
  /// The object schemas `node` makes the top-level object meet. `path` holds
  /// the references being followed.
  fn objects(&self, node: &'s Value, path: &mut Vec<&'s str>) -> Result<Vec<ObjectCaps>> {
    let Value::Object(map) = node else {
      return Ok(Vec::new());
    };
    let mut objects = Vec::new();
    if ["properties", "patternProperties", "additionalProperties"]
      .iter()
      .any(|keyword| map.contains_key(*keyword))
    {
      objects.push(self.object(map, path)?);
    }
    if let Some(reference) = map.get("$ref")
      && let Some((reference, target)) = self.resolve(reference, "$ref", path)?
    {
      path.push(reference);
      let referenced = self.objects(target, path);
      path.pop();
      objects.extend(referenced?);
    }
    if let Some(Value::Array(branches)) = map.get("allOf") {
      for branch in branches {
        objects.extend(self.objects(branch, path)?);
      }
    }
    for union in ["anyOf", "oneOf"] {
      if let Some(Value::Array(branches)) = map.get(union) {
        let mut agreed: Option<Vec<ObjectCaps>> = None;
        for branch in branches {
          if !self.may_hold(branch, Kind::Object, "$ref", path)? {
            continue;
          }
          let caps = self.objects(branch, path)?;
          match &agreed {
            None => agreed = Some(caps),
            Some(seen) => {
              if let Some(field) = disagreement(seen, &caps) {
                return Err(refusal(&field, BRANCH_DEPENDENT));
              }
            }
          }
        }
        objects.extend(agreed.unwrap_or_default());
      }
    }
    Ok(objects)
  }

  /// One object schema's caps: its `properties`, `patternProperties` and
  /// `additionalProperties`.
  fn object(&self, map: &'s Map<String, Value>, path: &mut Vec<&'s str>) -> Result<ObjectCaps> {
    let mut properties = BTreeMap::new();
    if let Some(Value::Object(declared)) = map.get("properties") {
      for (field, schema) in declared {
        properties.insert(field.clone(), self.string_cap(schema, field, path)?);
      }
    }
    let mut patterns = Vec::new();
    if let Some(Value::Object(declared)) = map.get("patternProperties") {
      for (pattern, schema) in declared {
        let field = format!("patternProperties[{pattern:?}]");
        let matcher =
          llmtask::Grammar::regex(pattern).map_err(|_| refusal(&field, UNMATCHABLE_PATTERN))?;
        patterns.push(PatternCap {
          pattern: pattern.clone(),
          matcher,
          cap: self.string_cap(schema, &field, path)?,
        });
      }
    }
    let additional = match map.get("additionalProperties") {
      Some(schema) => self.string_cap(schema, "additionalProperties", path)?,
      None => None,
    };
    Ok(ObjectCaps {
      properties,
      patterns,
      additional,
    })
  }

  /// The cap `node` puts on a string value of the member `field`.
  fn string_cap(
    &self,
    node: &'s Value,
    field: &str,
    path: &mut Vec<&'s str>,
  ) -> Result<Option<usize>> {
    // A boolean schema caps nothing.
    let Value::Object(map) = node else {
      return Ok(None);
    };
    let mut caps = vec![
      map
        .get("maxLength")
        .and_then(Value::as_u64)
        .and_then(|cap| usize::try_from(cap).ok()),
    ];
    if let Some(reference) = map.get("$ref")
      && let Some((reference, target)) = self.resolve(reference, field, path)?
    {
      path.push(reference);
      let referenced = self.string_cap(target, field, path);
      path.pop();
      caps.push(referenced?);
    }
    if let Some(Value::Array(branches)) = map.get("allOf") {
      for branch in branches {
        caps.push(self.string_cap(branch, field, path)?);
      }
    }
    for union in ["anyOf", "oneOf"] {
      if let Some(Value::Array(branches)) = map.get(union) {
        let mut agreed: Option<Option<usize>> = None;
        for branch in branches {
          if !self.may_hold(branch, Kind::String, field, path)? {
            continue;
          }
          let cap = self.string_cap(branch, field, path)?;
          match agreed {
            None => agreed = Some(cap),
            Some(seen) if seen == cap => {}
            Some(_) => return Err(refusal(field, BRANCH_DEPENDENT)),
          }
        }
        caps.push(agreed.flatten());
      }
    }
    Ok(caps.into_iter().flatten().min())
  }

  /// Whether `node` admits a value of `kind` at all: a branch whose `type`,
  /// `const` or `enum` rules the kind out cannot hold the value.
  fn may_hold(
    &self,
    node: &'s Value,
    kind: Kind,
    field: &str,
    path: &mut Vec<&'s str>,
  ) -> Result<bool> {
    let map = match node {
      Value::Bool(admits) => return Ok(*admits),
      Value::Object(map) => map,
      _ => return Ok(true),
    };
    let names_kind = match map.get("type") {
      None => true,
      Some(Value::String(name)) => name == kind.name(),
      Some(Value::Array(names)) => names.iter().any(|name| name == kind.name()),
      Some(_) => true,
    };
    if !names_kind
      || map.get("const").is_some_and(|value| !kind.holds(value))
      || map
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|options| !options.iter().any(|option| kind.holds(option)))
    {
      return Ok(false);
    }
    if let Some(reference) = map.get("$ref")
      && let Some((reference, target)) = self.resolve(reference, field, path)?
    {
      path.push(reference);
      let referenced = self.may_hold(target, kind, field, path);
      path.pop();
      if !referenced? {
        return Ok(false);
      }
    }
    if let Some(Value::Array(branches)) = map.get("allOf") {
      for branch in branches {
        if !self.may_hold(branch, kind, field, path)? {
          return Ok(false);
        }
      }
    }
    Ok(true)
  }

  /// The node `reference` points at, with the reference itself; `None` when
  /// it is met again on its own path, where it adds nothing.
  fn resolve(
    &self,
    reference: &'s Value,
    field: &str,
    path: &[&'s str],
  ) -> Result<Option<(&'s str, &'s Value)>> {
    let reference = reference
      .as_str()
      .ok_or_else(|| refusal(field, UNRESOLVED_REF))?;
    if path.contains(&reference) {
      return Ok(None);
    }
    self
      .target(reference)
      .map(|target| Some((reference, target)))
      .ok_or_else(|| refusal(field, UNRESOLVED_REF))
  }

  /// The node a local reference names: `#` (or `#/`) for the root, else the
  /// JSON Pointer after the `#` (serde_json unescapes `~1` and `~0`).
  fn target(&self, reference: &str) -> Option<&'s Value> {
    match reference.strip_prefix('#')? {
      "" | "/" => Some(self.root),
      pointer => self.root.pointer(pointer),
    }
  }
}

/// The first member two branches of a top-level `anyOf` or `oneOf` cap
/// differently, if any. Members both declare are compared by their caps; the
/// members a pattern or `additionalProperties` covers, by the caps those
/// declare.
fn disagreement(left: &[ObjectCaps], right: &[ObjectCaps]) -> Option<String> {
  if !left.iter().chain(right).any(ObjectCaps::caps_anything) {
    return None;
  }
  let declared: BTreeSet<&str> = left
    .iter()
    .chain(right)
    .flat_map(|object| object.properties.keys().map(String::as_str))
    .collect();
  if let Some(field) = declared
    .into_iter()
    .find(|field| cap_among(left, field) != cap_among(right, field))
  {
    return Some(field.to_owned());
  }
  let patterns = |objects: &[ObjectCaps]| -> BTreeSet<(String, Option<usize>)> {
    objects
      .iter()
      .flat_map(|object| {
        object
          .patterns
          .iter()
          .map(|pattern| (pattern.pattern.clone(), pattern.cap))
      })
      .collect()
  };
  if patterns(left) != patterns(right) {
    return Some("patternProperties".to_owned());
  }
  let additional = |objects: &[ObjectCaps]| -> Vec<Option<usize>> {
    objects.iter().map(|object| object.additional).collect()
  };
  (additional(left) != additional(right)).then(|| "additionalProperties".to_owned())
}

fn refusal(field: &str, reason: &'static str) -> Error {
  Error::UnsupportedFieldCap {
    field: SmolStr::new(field),
    reason,
  }
}

#[cfg(test)]
mod tests {
  use serde_json::json;

  use super::*;

  /// The cap `schema` resolves for its top-level member `field`.
  fn cap(schema: &Value, field: &str) -> Option<usize> {
    FieldCaps::of(schema)
      .unwrap_or_else(|e| panic!("{schema}: {e}"))
      .cap(field)
  }

  /// The member and reason `schema` is refused with.
  fn refused(schema: &Value) -> (String, &'static str) {
    match FieldCaps::of(schema) {
      Err(Error::UnsupportedFieldCap { field, reason }) => (field.to_string(), reason),
      other => panic!("{schema} must be refused, got {other:?}"),
    }
  }

  /// `{"type":"object","properties":{"description": description}}` with `rest`
  /// merged in at the top level.
  fn object(description: Value, rest: Value) -> Value {
    let mut schema = json!({"type": "object", "properties": {"description": description}});
    if let (Value::Object(schema), Value::Object(rest)) = (&mut schema, rest) {
      schema.extend(rest);
    }
    schema
  }

  /// LAW (Codex R2, [medium]): **a `$ref` caps its member as its target
  /// does**, through `#/$defs/…` and `#/definitions/…`, through a chain of
  /// references, and through a JSON Pointer with `~1` in it; a reference met
  /// again on its own path adds nothing, so a cycle still caps by what the
  /// cycle holds.
  #[test]
  fn a_ref_caps_its_member_as_its_target_does() {
    let defs = json!({
      "$defs": {
        "description": {"type": "string", "maxLength": 5},
        "a": {"$ref": "#/definitions/b"},
        "x/y": {"maxLength": 7},
        "loop": {"$ref": "#/$defs/back", "maxLength": 9},
        "back": {"$ref": "#/$defs/loop"},
        "spin": {"$ref": "#/$defs/spin"},
      },
      "definitions": {"b": {"type": "string", "maxLength": 4}},
    });
    for (reference, expected) in [
      ("#/$defs/description", Some(5)),
      ("#/$defs/a", Some(4)),
      ("#/$defs/x~1y", Some(7)),
      ("#/$defs/loop", Some(9)),
      ("#/$defs/spin", None),
    ] {
      let schema = object(json!({"$ref": reference}), defs.clone());
      assert_eq!(cap(&schema, "description"), expected, "{reference}");
    }
  }

  /// LAW (Codex R2, [medium]): **an `allOf` caps by its smallest
  /// `maxLength`**, the member's own included.
  #[test]
  fn all_of_caps_by_its_smallest() {
    let both = json!({"allOf": [{"type": "string", "maxLength": 8}, {"maxLength": 5}]});
    assert_eq!(cap(&object(both, json!({})), "description"), Some(5));
    let own = json!({"maxLength": 6, "allOf": [{"maxLength": 8}, {"type": "string"}]});
    assert_eq!(cap(&object(own, json!({})), "description"), Some(6));
    assert_eq!(
      cap(&object(json!({"type": "string"}), json!({})), "description"),
      None,
      "nothing caps it"
    );
  }

  /// LAW (Codex R2, [medium]): **a union whose branches disagree on the cap is
  /// refused by the member's name**: an `anyOf` of 5 and 8, a `oneOf` of which
  /// only one branch caps. Branches that agree cap as they do, and a branch
  /// that cannot hold a string (`null`, an integer, a `const` number) has no
  /// say.
  #[test]
  fn a_union_whose_branches_disagree_on_the_cap_is_refused_by_name() {
    for union in [
      json!({"anyOf": [{"type": "string", "maxLength": 5}, {"type": "string", "maxLength": 8}]}),
      json!({"oneOf": [{"type": "string", "maxLength": 5}, {"type": "string"}]}),
      json!({"allOf": [{"anyOf": [{"maxLength": 5}, {"maxLength": 8}]}]}),
    ] {
      assert_eq!(
        refused(&object(union.clone(), json!({}))),
        ("description".to_owned(), BRANCH_DEPENDENT),
        "{union}"
      );
    }
    for (union, expected) in [
      (
        json!({"anyOf": [{"maxLength": 5}, {"type": "string", "maxLength": 5}]}),
        Some(5),
      ),
      (
        json!({"anyOf": [{"type": "string", "maxLength": 5}, {"type": "null"}]}),
        Some(5),
      ),
      (
        json!({"oneOf": [{"type": "string", "maxLength": 5}, {"type": "integer"}, {"const": 3}]}),
        Some(5),
      ),
      (
        json!({"anyOf": [{"type": "string"}, {"type": ["null", "string"]}]}),
        None,
      ),
    ] {
      assert_eq!(
        cap(&object(union.clone(), json!({})), "description"),
        expected,
        "{union}"
      );
    }
  }

  /// LAW (Codex R2, [medium]): **a member `properties` does not declare is
  /// capped by `patternProperties` and `additionalProperties`**: by every
  /// pattern its key matches, anywhere in the key; by `additionalProperties`
  /// when none does; and a declared member that a pattern also matches meets
  /// both caps.
  #[test]
  fn an_undeclared_member_is_capped_by_patterns_and_additional_properties() {
    let schema = object(
      json!({"type": "string", "maxLength": 9}),
      json!({
        "patternProperties": {"^note": {"maxLength": 4}, "tion$": {"maxLength": 6}},
        "additionalProperties": {"type": "string", "maxLength": 3},
      }),
    );
    assert_eq!(cap(&schema, "notes"), Some(4));
    assert_eq!(cap(&schema, "notecaption"), Some(4), "both patterns match");
    assert_eq!(cap(&schema, "caption"), Some(6));
    assert_eq!(cap(&schema, "other"), Some(3));
    assert_eq!(
      cap(&schema, "description"),
      Some(6),
      "declared, and `tion$` matches"
    );
    let open = json!({"type": "object", "additionalProperties": true});
    assert_eq!(cap(&open, "anything"), None);
  }

  /// LAW (Codex R2, [medium]): **the top-level object meets its own `$ref`
  /// and `allOf` branches**, and a top-level union whose object branches cap a
  /// member differently is refused by that member's name; one whose
  /// branches agree, or whose other branches cannot be an object, is not.
  #[test]
  fn the_top_level_object_meets_its_refs_and_all_of() {
    let defs = json!({"$defs": {"answer": object(json!({"maxLength": 8}), json!({}))}});
    let mut by_ref = json!({"$ref": "#/$defs/answer"});
    by_ref
      .as_object_mut()
      .unwrap()
      .extend(defs.as_object().unwrap().clone());
    assert_eq!(cap(&by_ref, "description"), Some(8));
    let all_of = json!({"allOf": [
      object(json!({"maxLength": 8}), json!({})),
      object(json!({"maxLength": 5}), json!({})),
    ]});
    assert_eq!(cap(&all_of, "description"), Some(5));
    let disagree = json!({"anyOf": [
      object(json!({"maxLength": 8}), json!({})),
      object(json!({"maxLength": 5}), json!({})),
    ]});
    assert_eq!(
      refused(&disagree),
      ("description".to_owned(), BRANCH_DEPENDENT)
    );
    let agree = json!({"oneOf": [
      object(json!({"maxLength": 5}), json!({})),
      object(json!({"maxLength": 5}), json!({})),
      {"type": "null"},
    ]});
    assert_eq!(cap(&agree, "description"), Some(5));
  }

  /// LAW: **a reference the schema does not hold, and a pattern the `regex`
  /// crate cannot compile, are refused by name**, never read as no cap.
  #[test]
  fn an_unresolvable_ref_or_pattern_is_refused_by_name() {
    for reference in ["other.json#/x", "#/$defs/missing", "#nowhere"] {
      let schema = object(json!({"$ref": reference}), json!({"$defs": {}}));
      assert_eq!(
        refused(&schema),
        ("description".to_owned(), UNRESOLVED_REF),
        "{reference}"
      );
    }
    let schema = object(
      json!({"type": "string"}),
      json!({"patternProperties": {"(?<=a)b": {"maxLength": 3}}}),
    );
    let (field, reason) = refused(&schema);
    assert_eq!(reason, UNMATCHABLE_PATTERN);
    assert!(field.contains("(?<=a)b"), "{field}");
  }
}
