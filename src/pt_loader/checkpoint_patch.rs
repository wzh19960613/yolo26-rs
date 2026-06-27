//! Mutates parsed PyTorch checkpoint objects before re-serializing `data.pkl`.

mod object;

use std::collections::HashMap;

use candle_core::Tensor;
use candle_core::pickle::{Object, TensorInfo};

use crate::Result;

use object::*;

/// Class metadata written into the top-level Ultralytics model module.
pub(crate) struct ClassMetadata<'a> {
    pub(crate) labels_count: usize,
    pub(crate) names: Option<&'a [String]>,
}

/// Applies metadata and tensor-shape updates to every trainable module.
pub(crate) fn apply(
    root: &mut Object,
    dir_name: &str,
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&ClassMetadata<'_>>,
) -> Result<Vec<(String, TensorInfo)>> {
    for_each_top_module_mut(root, |module| {
        patch_module(module, "", tensors, metadata);
    });
    flatten_top_modules(root, dir_name)
}

fn flatten_top_modules(root: &Object, dir_name: &str) -> Result<Vec<(String, TensorInfo)>> {
    let mut out = Vec::new();
    if let Object::Dict(entries) = root {
        for (key, value) in entries {
            if is_top_module_key(key) {
                out.extend(super::flatten::flatten_module(value, dir_name, "")?);
            }
        }
    }
    if out.is_empty() {
        out.extend(super::flatten::flatten_module(root, dir_name, "model")?);
    }
    Ok(out)
}

fn for_each_top_module_mut(root: &mut Object, mut f: impl FnMut(&mut Object)) {
    match root {
        Object::Dict(entries) => {
            for (key, value) in entries {
                if is_top_module_key(key) {
                    f(value);
                }
            }
        }
        other => f(other),
    }
}

fn is_top_module_key(key: &Object) -> bool {
    matches!(key, Object::Unicode(name) if matches!(name.as_str(), "model" | "ema"))
}

fn patch_module(
    object: &mut Object,
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&ClassMetadata<'_>>,
) {
    let Some(state) = module_state_entries_mut(object) else {
        return;
    };
    patch_class_metadata(state, metadata);
    let weight_dims = patch_parameters(state, prefix, tensors);
    patch_channel_fields(state, weight_dims.as_deref());
    patch_children(state, prefix, tensors, metadata);
}

fn patch_class_metadata(state: &mut [(Object, Object)], metadata: Option<&ClassMetadata<'_>>) {
    let Some(metadata) = metadata else {
        return;
    };
    let old_nc = find_int(state, "nc");
    for (key, value) in state.iter_mut() {
        let Object::Unicode(name) = key else {
            continue;
        };
        match name.as_str() {
            "nc" => *value = usize_object(metadata.labels_count),
            "names" => *value = names_object(metadata),
            "no" => {
                if let (Some(old_nc), Some(old_no)) = (old_nc, object_to_usize(value)) {
                    let extra = old_no.saturating_sub(old_nc);
                    *value = usize_object(metadata.labels_count + extra);
                }
            }
            _ => {}
        }
    }
}

fn patch_parameters(
    state: &mut [(Object, Object)],
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
) -> Option<Vec<usize>> {
    let mut weight_dims = None;
    let Some(Object::Dict(parameters)) = state_value_mut(state, "_parameters") else {
        return None;
    };
    patch_leaf_tensors(parameters, prefix, tensors, &mut weight_dims);
    if let Some(Object::Dict(buffers)) = state_value_mut(state, "_buffers") {
        let mut ignored = None;
        patch_leaf_tensors(buffers, prefix, tensors, &mut ignored);
    }
    weight_dims
}

fn patch_leaf_tensors(
    leaves: &mut [(Object, Object)],
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    weight_dims: &mut Option<Vec<usize>>,
) {
    for (leaf_key, leaf_value) in leaves {
        let Object::Unicode(name) = leaf_key else {
            continue;
        };
        let full = join(prefix, name);
        let Some(tensor) = tensors.get(&full) else {
            continue;
        };
        let dims = tensor.dims().to_vec();
        patch_tensor_object(leaf_value, &dims);
        if name == "weight" {
            *weight_dims = Some(dims);
        }
    }
}

fn patch_children(
    state: &mut [(Object, Object)],
    prefix: &str,
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&ClassMetadata<'_>>,
) {
    let Some(Object::Dict(children)) = state_value_mut(state, "_modules") else {
        return;
    };
    for (child_key, child_value) in children {
        if let Object::Unicode(name) = child_key {
            patch_module(child_value, &join(prefix, name), tensors, metadata);
        }
    }
}

fn patch_channel_fields(state: &mut [(Object, Object)], dims: Option<&[usize]>) {
    let Some(dims) = dims else {
        return;
    };
    let groups = find_int(state, "groups").unwrap_or(1);
    let depthwise_groups = if dims.len() == 4 && groups > 1 && dims[1] == 1 {
        Some(dims[0])
    } else {
        None
    };
    for (key, value) in state {
        let Object::Unicode(name) = key else {
            continue;
        };
        match name.as_str() {
            "out_channels" | "out_features" if !dims.is_empty() => *value = usize_object(dims[0]),
            "in_channels" if let Some(groups) = depthwise_groups => *value = usize_object(groups),
            "in_channels" | "in_features" if dims.len() > 1 => *value = usize_object(dims[1]),
            "groups" if let Some(groups) = depthwise_groups => *value = usize_object(groups),
            _ => {}
        }
    }
}

fn patch_tensor_object(object: &mut Object, dims: &[usize]) {
    object::patch_tensor_object(object, dims);
}
