use crate::yoloe::prompt::session::{Session, ValidationRequest};

impl Session {
    /// Creates a validation request for the currently active prompt state.
    pub fn validation_request(&self) -> crate::Result<ValidationRequest> {
        self.ensure_ready_for_validation()?;
        Ok(ValidationRequest {
            usage: self.prompts.active_usage().ok_or_else(|| {
                crate::Error::InvalidConfig(
                    "YOLOE validation requires a non-empty prompt state".to_string(),
                )
            })?,
            classes: self.prompts.active_classes().ok_or_else(|| {
                crate::Error::InvalidConfig(
                    "YOLOE validation requires a non-empty prompt state".to_string(),
                )
            })?,
            predict: self.predict,
        })
    }

    pub(crate) fn ensure_ready_for_validation(&self) -> crate::Result<()> {
        if self.prompts.active_usage().is_none() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE validation requires text, visual, or prompt-free prompts".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn ensure_prompt_free_class_count(&self, expected: usize) -> crate::Result<()> {
        let classes = self.prompts.active_classes().ok_or_else(|| {
            crate::Error::InvalidConfig("YOLOE prompt-free vocabulary is empty".to_string())
        })?;
        if classes.len() != expected {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE prompt-free vocabulary has {} classes but LRPC head expects {expected}",
                classes.len()
            )));
        }
        Ok(())
    }
}
