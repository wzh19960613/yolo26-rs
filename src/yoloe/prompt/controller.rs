use crate::yoloe::config::Config;
use crate::yoloe::prompt::state::State;
use crate::yoloe::prompt::text_prompts::Text;
use crate::yoloe::prompt::visual::{Visual, visual_prompt_classes};
use crate::yoloe::usage::{EmbeddingSpace, FreeVocabulary, Usage};

/// YOLOE prompt controller.
#[derive(Debug, Clone)]
pub struct Controller {
    /// YOLOE configuration.
    pub config: Config,
    /// Current prompt state.
    pub state: State,
    /// Active prompt embedding-space descriptor.
    pub embedding_space: Option<EmbeddingSpace>,
}

impl Controller {
    /// Creates an empty prompt controller.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            state: State::Empty,
            embedding_space: None,
        }
    }

    /// Sets text class prompts.
    pub fn set_classes(&mut self, classes: Vec<String>) -> crate::Result<()> {
        self.ensure_mutable()?;
        self.config.validate_usage(Usage::TextPrompt)?;
        self.embedding_space = Some(EmbeddingSpace::new(
            self.config.prompt_dim,
            classes.clone(),
        )?);
        self.state = State::Text(Text::new(classes)?);
        Ok(())
    }

    /// Sets already-computed text prompt embeddings.
    pub fn set_prompt_embeddings(&mut self, space: EmbeddingSpace) -> crate::Result<()> {
        self.ensure_mutable()?;
        self.config.validate_usage(Usage::TextPrompt)?;
        if space.dim != self.config.prompt_dim {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE prompt embedding dim {} does not match config dim {}",
                space.dim, self.config.prompt_dim
            )));
        }
        self.state = State::Text(Text::new(space.classes.clone())?);
        self.embedding_space = Some(space);
        Ok(())
    }

    /// Sets visual prompts.
    pub fn set_visual_prompts(&mut self, prompts: Vec<Visual>) -> crate::Result<()> {
        self.ensure_mutable()?;
        self.config.validate_usage(Usage::Visual)?;
        if prompts.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompts require at least one exemplar".to_string(),
            ));
        }
        let classes = visual_prompt_classes(&prompts);
        self.embedding_space = Some(EmbeddingSpace::new(self.config.savpe.prompt_dim, classes)?);
        self.state = State::Visual(prompts);
        Ok(())
    }

    /// Sets prompt-free vocabulary.
    pub fn set_prompt_free_vocabulary(&mut self, classes: Vec<String>) -> crate::Result<()> {
        self.ensure_mutable()?;
        self.config.validate_usage(Usage::PromptFree)?;
        self.embedding_space = None;
        self.state = State::PromptFree(FreeVocabulary::new(classes)?);
        Ok(())
    }

    /// Returns the active prompt classes, if prompts are currently configured.
    pub fn active_classes(&self) -> Option<Vec<String>> {
        match &self.state {
            State::Empty => None,
            State::Text(prompts) => Some(prompts.classes.clone()),
            State::Visual(prompts) => Some(visual_prompt_classes(prompts)),
            State::PromptFree(vocabulary) => Some(vocabulary.classes.clone()),
        }
    }

    /// Returns the active prompt usage, if prompts are currently configured.
    pub fn active_usage(&self) -> Option<Usage> {
        match &self.state {
            State::Empty => None,
            State::Text(_) => Some(Usage::TextPrompt),
            State::Visual(_) => Some(Usage::Visual),
            State::PromptFree(_) => Some(Usage::PromptFree),
        }
    }

    fn ensure_mutable(&self) -> crate::Result<()> {
        Ok(())
    }
}
