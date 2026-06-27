//! Class-name validation for trainable model exports.

/// Validates optional class names against a model class count.
pub(crate) fn validate_class_names(
    labels_count: usize,
    names: Option<&[String]>,
) -> crate::Result<()> {
    let Some(names) = names else {
        return Ok(());
    };
    if names.len() != labels_count {
        return Err(crate::Error::InvalidConfig(format!(
            "class names count {} does not match labels_count {labels_count}",
            names.len()
        )));
    }
    if names.iter().any(|name| name.is_empty()) {
        return Err(crate::Error::InvalidConfig(
            "class names must not contain empty strings".to_string(),
        ));
    }
    Ok(())
}
