use crate::yoloe::prompt::text_prompts::Text;
use crate::yoloe::prompt::visual::Visual;
use crate::yoloe::usage::FreeVocabulary;

/// Prompt state attached to a YOLOE model.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum State {
    /// No prompts are active.
    #[default]
    Empty,
    /// Text class prompts are active.
    Text(Text),
    /// Visual exemplar prompts are active.
    Visual(Vec<Visual>),
    /// Prompt-free vocabulary is active.
    PromptFree(FreeVocabulary),
}
