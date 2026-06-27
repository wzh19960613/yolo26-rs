use crate::yoloe::config::Config;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::prompt::visual::Visual;
use crate::yoloe::reprta::RepRta;
use crate::yoloe::usage::EmbeddingTable;

impl Session {
    /// Creates a text-prompt session by encoding class names with the supplied
    /// [`ClipTextEncoder`](crate::yoloe::prompt::text_encoder::ClipTextEncoder)
    /// and aligning them through the model's RepRTA adapter when present
    /// (matching the official Python `set_classes` flow:
    /// CLIP → RepRTA → score).
    ///
    /// `encoder` is borrowed, so the same loaded CLIP model + tokenizer can be
    /// reused across many `Session::text` calls without reloading. `reprta` is
    /// borrowed from the [`Model`](crate::yoloe::segment::model::Model) via
    /// [`Model::reprta`](crate::yoloe::segment::model::Model::reprta); pass
    /// `None` when the checkpoint has no RepRTA. Class names accept anything
    /// `AsRef<str>` (`&str`, `String`, `&&str`, ...), so callers never need to
    /// write `.into()` per element.
    ///
    /// Requires the `yoloe-text` feature (included in the `yoloe` aggregate).
    #[cfg(feature = "yoloe-text")]
    pub fn text<I, S>(
        encoder: &crate::yoloe::prompt::text_encoder::ClipTextEncoder,
        reprta: Option<&RepRta>,
        classes: I,
    ) -> crate::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::text_with_config(encoder, reprta, classes, Config::default())
    }

    /// Like [`Self::text`] but uses an explicit [`Config`].
    ///
    /// Requires the `yoloe-text` feature (included in the `yoloe` aggregate).
    #[cfg(feature = "yoloe-text")]
    pub fn text_with_config<I, S>(
        encoder: &crate::yoloe::prompt::text_encoder::ClipTextEncoder,
        reprta: Option<&RepRta>,
        classes: I,
        config: Config,
    ) -> crate::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut session = Self::new(config);
        session.set_classes_with_clip_embeddings(encoder, reprta, classes)?;
        Ok(session)
    }

    /// Creates a text-prompt session from already-computed (e.g. externally
    /// encoded) embeddings. Available whenever any `yoloe-*` feature is on.
    pub fn text_with_embeddings(table: EmbeddingTable) -> crate::Result<Self> {
        Self::text_with_embeddings_and_config(table, Config::default())
    }

    /// Like [`Self::text_with_embeddings`] but uses an explicit [`Config`].
    pub fn text_with_embeddings_and_config(
        table: EmbeddingTable,
        config: Config,
    ) -> crate::Result<Self> {
        let mut session = Self::new(config);
        session.set_text_prompt_embeddings(table)?;
        Ok(session)
    }

    /// Loads a RepRTA adapter and creates a text-prompt session from
    /// already-aligned embeddings, auto-applying RepRTA when
    /// `config.rep_rta.enabled`.
    pub fn text_with_reprta(
        reprta: &RepRta,
        table: EmbeddingTable,
        config: Config,
    ) -> crate::Result<Self> {
        let mut session = Self::new(config);
        session.set_text_prompt_embeddings_with_reprta(reprta, table)?;
        Ok(session)
    }

    /// Creates a prompt-free session for a static class vocabulary.
    pub fn prompt_free(classes: Vec<String>) -> crate::Result<Self> {
        Self::prompt_free_with_config(classes, Config::default())
    }

    /// Like [`Self::prompt_free`] but uses an explicit [`Config`]. The config
    /// is adjusted to the prompt-free checkpoint family.
    pub fn prompt_free_with_config(
        classes: Vec<String>,
        mut config: Config,
    ) -> crate::Result<Self> {
        if !config.prompt_free {
            config.prompt_free = true;
            config.lrpc = true;
        }
        let mut session = Self::new(config);
        session.set_prompt_free_vocabulary(classes)?;
        Ok(session)
    }

    /// Creates a prompt-free session from static LRPC embeddings.
    pub fn prompt_free_with_embeddings(table: EmbeddingTable) -> crate::Result<Self> {
        let mut session = Self::new(Config::default());
        session.set_prompt_free_embeddings(table)?;
        Ok(session)
    }

    /// Creates a prompt-free session using the built-in LRPC vocabulary names
    /// (requires the `yoloe-pf` feature).
    ///
    /// The 4585 display names from `LRPC_VOCAB` are attached verbatim; scoring
    /// still comes from the checkpoint's `vocab.weight`. This is the convenient
    /// counterpart of [`Self::prompt_free`](Self::prompt_free) when the caller
    /// wants readable class names (`person`, `bus`, ...) instead of `pf_XXXX`
    /// placeholders and does not maintain its own 4585-entry mapping.
    #[cfg(feature = "yoloe-pf")]
    pub fn prompt_free_default() -> crate::Result<Self> {
        Self::prompt_free(
            crate::default_labels::LRPC_VOCAB
                .iter()
                .map(|s| s.to_string())
                .collect(),
        )
    }

    /// Creates a visual-prompt session from source-image prompts.
    ///
    /// Visual prompts are image-specific: build a fresh session per image and
    /// pass it to `predict_visual_prompts`. The box/mask distinction is made at
    /// predict time via [`VisualSource`](crate::yoloe::visuals::VisualSource),
    /// so session construction does not need to know the source form. SAVPE
    /// embeddings are computed during the forward pass (they require the
    /// backbone features), so this session does not carry a precomputed table.
    pub fn visual(prompts: Vec<Visual>) -> crate::Result<Self> {
        Self::visual_with_config(prompts, Config::default())
    }

    /// Like [`Self::visual`] but uses an explicit [`Config`].
    pub fn visual_with_config(prompts: Vec<Visual>, config: Config) -> crate::Result<Self> {
        let mut session = Self::new(config);
        session.set_visual_prompts(prompts)?;
        Ok(session)
    }
}
