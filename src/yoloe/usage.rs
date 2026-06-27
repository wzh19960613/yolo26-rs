use candle_core::Tensor;

/// YOLOE usage family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum Usage {
    /// Text-prompt open-vocabulary inference.
    TextPrompt,
    /// Visual-prompt inference from exemplar images or regions.
    Visual,
    /// Prompt-free inference with a fixed vocabulary.
    PromptFree,
    /// Fine-tuning the YOLOE model.
    FineTune,
    /// Linear probing on top of frozen YOLOE features.
    LinearProbe,
    /// Validation or benchmarking.
    Validate,
}

impl Usage {
    /// Returns a stable lowercase usage token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextPrompt => "text_prompt",
            Self::Visual => "visual_prompt",
            Self::PromptFree => "prompt_free",
            Self::FineTune => "fine_tune",
            Self::LinearProbe => "linear_probe",
            Self::Validate => "validate",
        }
    }
}

/// YOLOE checkpoint family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointKind {
    /// Text-prompt or visual-prompt YOLOE checkpoint.
    #[default]
    Prompted,
    /// Prompt-free YOLOE checkpoint with static vocabulary.
    PromptFree,
    /// Segmentation-first YOLOE checkpoint.
    Segmentation,
}

/// Base task family used by a YOLOE checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum BaseTask {
    /// Detection-head model initialized from a YOLOE YAML.
    Detect,
    /// Official YOLOE segmentation-first checkpoint.
    Segment,
}

/// Re-parameterizable Region-Text Alignment module settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub struct RepRtaConfig {
    /// Whether RepRTA text alignment is present.
    pub enabled: bool,
    /// Whether RepRTA has been folded into the inference head.
    pub folded_for_inference: bool,
}

impl Default for RepRtaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            folded_for_inference: false,
        }
    }
}

/// Semantic-Activated Visual Prompt Encoder settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub struct SavpeConfig {
    /// Whether visual prompt encoding is present.
    pub enabled: bool,
    /// Visual prompt embedding dimensions.
    pub prompt_dim: usize,
}

impl Default for SavpeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prompt_dim: 512,
        }
    }
}

/// Lazy Region-Prompt Contrast prompt-free vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]

pub struct FreeVocabulary {
    /// Static class vocabulary used for prompt-free inference.
    pub classes: Vec<String>,
}

impl FreeVocabulary {
    /// Creates a prompt-free vocabulary.
    pub fn new(classes: Vec<String>) -> crate::Result<Self> {
        if classes.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE LRPC vocabulary must not be empty".to_string(),
            ));
        }
        Ok(Self { classes })
    }
}

/// Prompt embedding space associated with active text or visual prompts.
#[derive(Debug, Clone, PartialEq, Eq)]

pub struct EmbeddingSpace {
    /// Embedding dimension.
    pub dim: usize,
    /// Class labels represented by the embedding rows.
    pub classes: Vec<String>,
}

impl EmbeddingSpace {
    /// Creates a prompt embedding-space descriptor.
    pub fn new(dim: usize, classes: Vec<String>) -> crate::Result<Self> {
        if dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt embedding dimension must be greater than zero".to_string(),
            ));
        }
        if classes.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt embedding space requires at least one class".to_string(),
            ));
        }
        Ok(Self { dim, classes })
    }
}

/// Prompt embedding tensor used to score dense YOLOE region features.
#[derive(Debug, Clone)]

pub struct EmbeddingTable {
    /// Class labels represented by embedding rows.
    pub classes: Vec<String>,
    /// Prompt embeddings with shape `[classes, dim]`.
    pub embeddings: Tensor,
}
