//! Recursively flattens a PyTorch `nn.Module` object graph into a flat list of
//! dotted-name tensor infos, mirroring `module.state_dict()`.
//!
//! An `nn.Module` is serialized as a `BUILD` over a `__setstate__` dict whose
//! keys include `_parameters`, `_buffers` and `_modules`. Each child module in
//! `_modules` contributes its dotted prefix; each parameter/buffer contributes
//! a leaf tensor info via [`candle_core::pickle::Object::into_tensor_info`].

use std::path::Path;

use candle_core::pickle::{Object, TensorInfo};

use crate::Result;

/// Flattens a module object into `(dotted_name, TensorInfo)` pairs. The root
/// object may be a module (`BUILD`/`REDUCE`), a state-dict-like `Dict`, or a
/// plain value; unsupported branches are silently skipped. `root_prefix` names
/// the root module (e.g. `model`) so keys match the safetensors layout.
pub fn flatten_module(
    root: &Object,
    dir_name: &str,
    root_prefix: &str,
) -> Result<Vec<(String, TensorInfo)>> {
    let mut out = Vec::new();
    walk(root, dir_name, root_prefix, &mut out)?;
    Ok(out)
}

fn walk(
    object: &Object,
    dir_name: &str,
    prefix: &str,
    out: &mut Vec<(String, TensorInfo)>,
) -> Result<()> {
    if let Some(state) = module_state_dict(object) {
        walk_module_state(state, dir_name, prefix, out)?;
        return Ok(());
    }
    // A plain dict that already looks like a state_dict (parameter name -> tensor).
    if let Object::Dict(entries) = object {
        for (key, value) in entries {
            let name = match key {
                Object::Unicode(s) => s.clone(),
                _ => continue,
            };
            push_leaf(value, dir_name, prefix, &name, out);
        }
    }
    Ok(())
}

/// Returns the module's `__setstate__` dict if `object` is a serialized module.
fn module_state_dict(object: &Object) -> Option<Vec<(Object, Object)>> {
    let args = match object {
        Object::Build { callable, args } => {
            let inner = match &**callable {
                Object::Reduce { callable, .. } => &**callable,
                _ => return None,
            };
            // Only treat REDUCE-over-Class (arbitrary nn.Module subclass) builds
            // as module containers; other builds (e.g. plain tensors) are leaves.
            match inner {
                Object::Class { .. } => &**args,
                _ => return None,
            }
        }
        Object::Reduce { callable, args } => match &**callable {
            Object::Class { .. } => &**args,
            _ => return None,
        },
        _ => return None,
    };
    match args {
        Object::Dict(state) => Some(state.clone()),
        _ => None,
    }
}

fn walk_module_state(
    state: Vec<(Object, Object)>,
    dir_name: &str,
    prefix: &str,
    out: &mut Vec<(String, TensorInfo)>,
) -> Result<()> {
    for (key, value) in &state {
        let field = match key {
            Object::Unicode(s) => s.as_str(),
            _ => continue,
        };
        match field {
            "_parameters" | "_buffers" => {
                if let Object::Dict(leaves) = value {
                    for (leaf_key, leaf_val) in leaves {
                        if let Object::Unicode(name) = leaf_key {
                            push_leaf(leaf_val, dir_name, prefix, name, out);
                        }
                    }
                }
            }
            "_modules" => {
                if let Object::Dict(children) = value {
                    for (child_key, child_val) in children {
                        if let Object::Unicode(name) = child_key {
                            let next = join(prefix, name);
                            walk(child_val, dir_name, &next, out)?;
                        }
                    }
                }
            }
            "stride" => push_leaf(value, dir_name, prefix, field, out),
            _ => {}
        }
    }
    Ok(())
}

fn push_leaf(
    value: &Object,
    dir_name: &str,
    prefix: &str,
    name: &str,
    out: &mut Vec<(String, TensorInfo)>,
) {
    let full = join(prefix, name);
    let name_obj = Object::Unicode(name.to_string());
    if let Ok(Some(info)) = value
        .clone()
        .into_tensor_info(name_obj, Path::new(dir_name))
    {
        out.push((full, info));
    }
}

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
