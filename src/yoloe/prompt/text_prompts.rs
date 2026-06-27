/// Text prompt class vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text {
    /// Class names in the prompt vocabulary.
    pub classes: Vec<String>,
}

impl Text {
    /// Creates text prompts from class names.
    pub fn new(classes: Vec<String>) -> crate::Result<Self> {
        if classes.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE text prompts require at least one class".to_string(),
            ));
        }
        Ok(Self { classes })
    }
}
