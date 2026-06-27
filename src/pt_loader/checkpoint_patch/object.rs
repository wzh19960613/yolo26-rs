//! Low-level helpers for editing Candle pickle objects.

use candle_core::pickle::Object;

use super::ClassMetadata;

pub(super) fn patch_tensor_object(object: &mut Object, dims: &[usize]) {
    match object {
        Object::Reduce { callable, args }
            if is_class(callable, "torch._utils", "_rebuild_tensor_v2") =>
        {
            patch_rebuild_tensor_args(args, dims);
        }
        Object::Reduce { callable, args }
            if is_class(callable, "torch._utils", "_rebuild_parameter") =>
        {
            if let Object::Tuple(items) = args.as_mut()
                && let Some(tensor) = items.first_mut()
            {
                patch_tensor_object(tensor, dims);
            }
        }
        Object::Reduce { callable, args }
            if is_class(callable, "torch._tensor", "_rebuild_from_type_v2") =>
        {
            if let Object::Tuple(items) = args.as_mut()
                && let Some(tensor_args) = items.get_mut(2)
            {
                patch_rebuild_tensor_args(tensor_args, dims);
            }
        }
        Object::Build { callable, args } | Object::Reduce { callable, args } => {
            patch_tensor_object(callable, dims);
            patch_tensor_object(args, dims);
        }
        _ => {}
    }
}

pub(super) fn module_state_entries_mut(object: &mut Object) -> Option<&mut Vec<(Object, Object)>> {
    let args = match object {
        Object::Build { callable, args } => {
            let inner = match callable.as_ref() {
                Object::Reduce { callable, .. } => callable.as_ref(),
                _ => return None,
            };
            match inner {
                Object::Class { .. } => args.as_mut(),
                _ => return None,
            }
        }
        Object::Reduce { callable, args } => match callable.as_ref() {
            Object::Class { .. } => args.as_mut(),
            _ => return None,
        },
        _ => return None,
    };
    match args {
        Object::Dict(state) => Some(state),
        _ => None,
    }
}

pub(super) fn state_value_mut<'a>(
    state: &'a mut [(Object, Object)],
    name: &str,
) -> Option<&'a mut Object> {
    state
        .iter_mut()
        .find(|(key, _)| matches!(key, Object::Unicode(key) if key == name))
        .map(|(_, value)| value)
}

pub(super) fn find_int(state: &[(Object, Object)], name: &str) -> Option<usize> {
    state
        .iter()
        .find(|(key, _)| matches!(key, Object::Unicode(key) if key == name))
        .and_then(|(_, value)| object_to_usize(value))
}

pub(super) fn object_to_usize(object: &Object) -> Option<usize> {
    match object {
        Object::Int(value) if *value >= 0 => Some(*value as usize),
        Object::Long(value) if *value >= 0 => Some(*value as usize),
        _ => None,
    }
}

pub(super) fn names_object(metadata: &ClassMetadata<'_>) -> Object {
    let names = metadata
        .names
        .map(|names| names.to_vec())
        .unwrap_or_else(|| {
            (0..metadata.labels_count)
                .map(|idx| format!("class_{idx}"))
                .collect()
        });
    Object::Dict(
        names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| (usize_object(idx), Object::Unicode(name)))
            .collect(),
    )
}

pub(super) fn usize_object(value: usize) -> Object {
    i32::try_from(value).map_or(Object::Long(value as i64), Object::Int)
}

pub(super) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

fn patch_rebuild_tensor_args(args: &mut Object, dims: &[usize]) {
    let Object::Tuple(items) = args else {
        return;
    };
    if items.len() < 4 {
        return;
    }
    let offset = object_to_usize(&items[1]).unwrap_or(0);
    items[2] = usize_tuple(dims);
    items[3] = usize_tuple(&contiguous_stride(dims));
    if let Object::PersistentLoad(storage) = &mut items[0] {
        patch_storage_size(storage, offset + dims.iter().product::<usize>());
    }
}

fn patch_storage_size(storage: &mut Object, size: usize) {
    if let Object::Tuple(items) = storage
        && items.len() > 4
    {
        items[4] = usize_object(size);
    }
}

fn usize_tuple(values: &[usize]) -> Object {
    Object::Tuple(values.iter().copied().map(usize_object).collect())
}

fn contiguous_stride(dims: &[usize]) -> Vec<usize> {
    let mut stride = vec![1; dims.len()];
    for idx in (1..dims.len()).rev() {
        stride[idx - 1] = stride[idx] * dims[idx];
    }
    stride
}

fn is_class(object: &Object, module: &str, class: &str) -> bool {
    matches!(
        object,
        Object::Class {
            module_name,
            class_name,
        } if module_name == module && class_name == class
    )
}
