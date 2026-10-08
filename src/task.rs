//! Re-exports of the `llmtask` Task trait for convenience, with the
//! decoder's account `Task::parse_ended` takes.
//!
//! Users can `use lfm::Task` instead of `use llmtask::Task`.

pub use llmtask::{FieldEnd, FieldEnds, JsonParseError, Task};
