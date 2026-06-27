use std::collections::BTreeMap;

pub(crate) fn first_dim(shapes: &BTreeMap<String, Vec<usize>>, key: &str) -> crate::Result<usize> {
    let shape = shapes
        .get(key)
        .ok_or_else(|| crate::Error::InvalidConfig(format!("missing tensor shape for {key}")))?;
    shape
        .first()
        .copied()
        .filter(|dim| *dim > 0)
        .ok_or_else(|| {
            crate::Error::InvalidTensor(format!(
                "tensor {key} must have non-empty shape, got {shape:?}"
            ))
        })
}
