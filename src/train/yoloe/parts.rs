//! Trainable YOLOE parameter family for a training mode.

/// Trainable YOLOE parameter family for a training mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parts {
    /// Train every configured YOLOE parameter.
    All,
    /// Train only the final classification projections in `cv3`/`one2one_cv3`.
    ClassificationBranch,
    /// Train only the SAVPE visual-prompt module.
    Savpe,
    /// Train only the prompt-free classifier branch.
    PromptFreeClassifier,
}

impl Parts {
    /// Returns true when a variable name is trainable for this recipe part.
    pub fn allows_variable(self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::ClassificationBranch | Self::PromptFreeClassifier => {
                is_final_cv3_projection(name)
            }
            Self::Savpe => name.contains(".savpe."),
        }
    }
}

/// Returns true when `name` is a final `cv3`/`one2one_cv3` projection weight.
fn is_final_cv3_projection(name: &str) -> bool {
    name.starts_with("model.23.")
        && (name.contains(".cv3.") || name.contains(".one2one_cv3."))
        && (name.contains(".0.2.") || name.contains(".1.2.") || name.contains(".2.2."))
}
