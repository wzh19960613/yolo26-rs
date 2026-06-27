//! Lightweight metadata extraction from a parsed checkpoint: class names,
//! epoch and best-fitness, matching the fields Ultralytics writes into the
//! top-level dict.

use candle_core::pickle::Object;

use crate::Result;

use super::reader::ParsedCheckpoint;

/// Class-name table carried by the checkpoint (`{id: name}`), the same shape as
/// the official `model.names`. Names are returned in id order as a `Vec`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PtNames(pub Vec<String>);

/// Identity metadata useful for picking a config (class count) and for
/// reporting resume state.
#[derive(Debug, Clone, Default)]
pub struct PtCheckpointMetadata {
    /// Number of training epochs already completed (`-1` for a fresh pretrained
    /// checkpoint), or `None` if the field is absent.
    pub epoch: Option<i64>,
    /// Best-fitness value recorded so far, or `None`.
    pub best_fitness: Option<f64>,
    /// Ultralytics version string, or `None`.
    pub version: Option<String>,
    /// Class names in id order, or `None` when not present.
    pub names: Option<PtNames>,
}

/// Extracts metadata from a parsed checkpoint's shallow top-level dict.
pub fn extract(parsed: &ParsedCheckpoint) -> Result<PtCheckpointMetadata> {
    let entries = match &parsed.top_level {
        Object::Dict(entries) => entries,
        _ => return Ok(PtCheckpointMetadata::default()),
    };
    let get = |key: &str| -> Option<&Object> {
        entries
            .iter()
            .find(|(k, _)| matches!(k, Object::Unicode(s) if s == key))
            .map(|(_, v)| v)
    };
    // `names` lives on the module state dict; fall back to the top level.
    let names = lookup(&parsed.module_state, "names").or_else(|| get("names"));
    let names = names.and_then(extract_names);
    Ok(PtCheckpointMetadata {
        epoch: get("epoch").and_then(|v| v.clone().int_or_long().ok()),
        best_fitness: get("best_fitness").and_then(|v| match v {
            Object::Float(x) => Some(*x),
            Object::Int(x) => Some(*x as f64),
            Object::Long(x) => Some(*x as f64),
            _ => None,
        }),
        version: get("version").and_then(|v| v.clone().unicode().ok()),
        names,
    })
}

fn lookup<'a>(dict: &'a Object, key: &str) -> Option<&'a Object> {
    match dict {
        Object::Dict(entries) => entries
            .iter()
            .find(|(k, _)| matches!(k, Object::Unicode(s) if s == key))
            .map(|(_, v)| v),
        _ => None,
    }
}

/// Converts the `names` field into an id-ordered `PtNames`. Ultralytics stores
/// names as a `{id: name}` dict or a `["name", ...]` list.
fn extract_names(value: &Object) -> Option<PtNames> {
    match value {
        Object::Dict(pairs) => {
            let mut indexed: Vec<(i64, String)> = pairs
                .iter()
                .filter_map(|(k, v)| match (k, v) {
                    (Object::Long(id), Object::Unicode(name)) => Some((*id, name.clone())),
                    (Object::Int(id), Object::Unicode(name)) => Some((*id as i64, name.clone())),
                    _ => None,
                })
                .collect();
            indexed.sort_unstable_by_key(|(id, _)| *id);
            if indexed.is_empty() {
                None
            } else {
                Some(PtNames(indexed.into_iter().map(|(_, n)| n).collect()))
            }
        }
        Object::List(items) => {
            let names: Vec<String> = items
                .iter()
                .filter_map(|v| v.clone().unicode().ok())
                .collect();
            (!names.is_empty()).then_some(PtNames(names))
        }
        _ => None,
    }
}
