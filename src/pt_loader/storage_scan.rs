//! Whole-graph storage scan for template-based `.pt` writing.
//!
//! [`super::flatten::flatten_module`] only walks the `model` subtree, but a
//! `data.pkl` template references every storage of the whole checkpoint
//! (model + EMA + optimizer + misc buffers). To regenerate a `torch.load`-safe
//! zip we must emit a blob for *every* storage the pickle references, otherwise
//! `torch.load` fails on a missing `data/<N>` entry.
//!
//! This module walks the full deserialized object graph and turns every tensor
//! reduction (`_rebuild_tensor_v2` / `_rebuild_parameter` / `_rebuild_from_type_v2`)
//! into a [`TensorInfo`], deduplicated by storage path so each `data/<N>` blob is
//! written exactly once.

use std::collections::HashSet;
use std::path::Path;

use candle_core::pickle::{Object, TensorInfo};

use crate::Result;

/// Collects every tensor/storage reference reachable from `root`, deduplicated
/// by blob path (`<dir>/data/<N>`). `dir_name` is the archive root that prefixes
/// storage paths (e.g. `best/data`), matching how candle builds `TensorInfo.path`.
///
/// Names are synthetic (just the storage path); only the layout/dtype/path
/// matter for blob generation.
pub(crate) fn collect_all(root: &Object, dir_name: &str) -> Result<Vec<TensorInfo>> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out: Vec<TensorInfo> = Vec::new();
    visit(root, dir_name, &mut seen, &mut out);
    Ok(out)
}

/// Recursively visits container objects and pushes any tensor reduction it finds.
fn visit(object: &Object, dir_name: &str, seen: &mut HashSet<String>, out: &mut Vec<TensorInfo>) {
    match object {
        // Tensors resolve to a TensorInfo; everything else is either a leaf or a
        // container we keep walking. Cloning is cheap relative to the pkl size and
        // required because `into_tensor_info` consumes the object.
        Object::Reduce { .. } | Object::Build { .. } => {
            if let Some(info) = try_tensor_info(object.clone(), dir_name)
                && seen.insert(info.path.clone())
            {
                out.push(info);
            }
            // Build wraps a Reduce; keep descending in case nested tensors exist.
            descend(object, dir_name, seen, out);
        }
        Object::Tuple(items) | Object::List(items) => {
            for item in items {
                visit(item, dir_name, seen, out);
            }
        }
        Object::Dict(pairs) => {
            for (k, v) in pairs {
                visit(k, dir_name, seen, out);
                visit(v, dir_name, seen, out);
            }
        }
        Object::PersistentLoad(inner) => visit(inner, dir_name, seen, out),
        _ => {}
    }
}

/// Descends into Reduce/Build children to catch tensors nested inside them.
fn descend(object: &Object, dir_name: &str, seen: &mut HashSet<String>, out: &mut Vec<TensorInfo>) {
    let children: Vec<&Object> = match object {
        Object::Reduce { callable, args } | Object::Build { callable, args } => {
            vec![callable.as_ref(), args.as_ref()]
        }
        _ => return,
    };
    for child in children {
        visit(child, dir_name, seen, out);
    }
}

/// Tries to interpret `object` as a tensor reduction, mirroring candle's
/// `Object::into_tensor_info` with a synthetic name.
fn try_tensor_info(object: Object, dir_name: &str) -> Option<TensorInfo> {
    let name = Object::Unicode(String::new());
    object
        .into_tensor_info(name, Path::new(dir_name))
        .ok()
        .flatten()
}
