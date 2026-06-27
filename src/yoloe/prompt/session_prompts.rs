use candle_core::Tensor;

use crate::yoloe::checkpoint::identity::Identity;
use crate::yoloe::config::Config;
use crate::yoloe::head::contrastive::Contrastive;
use crate::yoloe::predict_config::PredictConfig;
use crate::yoloe::prompt::controller::Controller;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::prompt::table::ScorerConfig;
use crate::yoloe::prompt::visual::{Visual, visual_prompt_classes};
use crate::yoloe::reprta::RepRta;
use crate::yoloe::savpe::pooler::Pooler;
use crate::yoloe::usage::{EmbeddingTable, Usage};

impl Session {
    /// Creates a YOLOE session from an explicit configuration.
    pub fn new(config: Config) -> Self {
        Self {
            checkpoint: None,
            prompts: Controller::new(config),
            predict: PredictConfig::default(),
            scorer: ScorerConfig::default(),
            prompt_table: None,
            reprta: None,
        }
    }

    /// Creates a YOLOE session from an official-style checkpoint name.
    pub fn from_checkpoint(name: impl AsRef<str>) -> crate::Result<Self> {
        let checkpoint = Identity::parse(name)?;
        let mut session = Self::new(checkpoint.config());
        session.checkpoint = Some(checkpoint);
        Ok(session)
    }

    /// Returns the YOLOE model configuration.
    pub fn config(&self) -> &Config {
        &self.prompts.config
    }

    /// Returns the active prompt embedding table, if one has been supplied.
    pub fn prompt_table(&self) -> Option<&EmbeddingTable> {
        self.prompt_table.as_ref()
    }

    /// Sets text-prompt class names without precomputed embeddings.
    ///
    /// **This records the prompt state only; it does NOT produce embeddings.**
    /// Until a prompt embedding table is supplied, text-prompt scoring returns
    /// an error. Activate a table via one of:
    /// - [`Self::set_classes_with_clip_embeddings`] (CLIP via mobileclip2-b-rs),
    /// - [`Self::set_text_prompt_embeddings`] (external/official CLIP table),
    /// - [`Self::set_text_prompt_embeddings_with_reprta`] (official CLIP + RepRTA).
    pub fn set_classes(&mut self, classes: Vec<String>) -> crate::Result<()> {
        self.prompts.set_classes(classes)?;
        self.prompt_table = None;
        Ok(())
    }

    /// Sets text-prompt class names by encoding them with the supplied
    /// [`ClipTextEncoder`](crate::yoloe::prompt::text_encoder::ClipTextEncoder)
    /// and aligning through the model's RepRTA adapter when supplied.
    ///
    /// This is the official text-prompt path (CLIP → RepRTA → score):
    /// - `encoder` is borrowed and may be reused across many sessions.
    /// - `classes` accepts any `AsRef<str>` (`&str`, `String`, ...).
    /// - `reprta` is borrowed from the
    ///   [`Model`](crate::yoloe::segment::model::Model) via
    ///   [`Model::reprta`](crate::yoloe::segment::model::Model::reprta); when
    ///   `Some` and `config.rep_rta.enabled` (the default) the embeddings are
    ///   aligned, otherwise they are activated as-is.
    ///
    /// The resulting `[classes, 512]` embeddings are L2-normalized and placed
    /// on the CPU; move the returned prompt table to another device before GPU
    /// scoring.
    #[cfg(feature = "yoloe-text")]
    pub fn set_classes_with_clip_embeddings<I, S>(
        &mut self,
        encoder: &crate::yoloe::prompt::text_encoder::ClipTextEncoder,
        reprta: Option<&RepRta>,
        classes: I,
    ) -> crate::Result<()>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Collect names first (the encoder consumes the iterator).
        let names: Vec<String> = classes.into_iter().map(|s| s.as_ref().to_owned()).collect();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let embeddings = encoder.embed_classes(&refs)?;

        let table = EmbeddingTable::new(embeddings, names)?;
        match (reprta, self.prompts.config.rep_rta.enabled) {
            (Some(reprta), true) => {
                self.set_text_prompt_embeddings_with_reprta(reprta, table)?;
            }
            _ => {
                self.set_text_prompt_embeddings(table)?;
            }
        }
        Ok(())
    }

    /// Loads a RepRTA text-prompt adapter so that subsequent text-prompt
    /// embeddings are auto-aligned (matching official inference, where RepRTA is
    /// applied by default to RepRTA-enabled checkpoints).
    pub fn set_reprta(&mut self, reprta: RepRta) {
        self.reprta = Some(reprta);
    }

    /// Sets text-prompt embeddings and the corresponding class names.
    ///
    /// When a RepRTA adapter is loaded and `config.rep_rta.enabled` (the default),
    /// the embeddings are auto-aligned through RepRTA before activation, matching
    /// the official text-prompt inference path.
    pub fn set_text_prompt_embeddings(&mut self, table: EmbeddingTable) -> crate::Result<()> {
        let aligned = match (&self.reprta, self.prompts.config.rep_rta.enabled) {
            (Some(reprta), true) => reprta.forward_table(&table)?,
            _ => table,
        };
        self.prompts
            .set_prompt_embeddings(aligned.embedding_space()?)?;
        self.prompt_table = Some(aligned);
        Ok(())
    }

    /// Loads a RepRTA adapter and activates already-aligned text-prompt
    /// embeddings.
    ///
    /// Equivalent to [`Self::set_reprta`] followed by
    /// [`Self::set_text_prompt_embeddings`]; the adapter is stored so later
    /// text-prompt embeddings are auto-aligned too.
    pub fn set_text_prompt_embeddings_with_reprta(
        &mut self,
        reprta: &RepRta,
        table: EmbeddingTable,
    ) -> crate::Result<()> {
        self.reprta = Some(reprta.clone());
        self.set_text_prompt_embeddings(table)
    }

    /// Sets visual prompt regions without precomputed SAVPE embeddings.
    pub fn set_visual_prompts(&mut self, prompts: Vec<Visual>) -> crate::Result<()> {
        self.prompts.set_visual_prompts(prompts)?;
        self.prompt_table = None;
        Ok(())
    }

    /// Sets visual prompt regions together with precomputed SAVPE embeddings.
    pub fn set_visual_prompt_embeddings(
        &mut self,
        prompts: Vec<Visual>,
        table: EmbeddingTable,
    ) -> crate::Result<()> {
        let classes = visual_prompt_classes(&prompts);
        if table.classes != classes {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE visual prompt embedding classes {:?} do not match prompt classes {:?}",
                table.classes, classes
            )));
        }
        self.prompts.set_visual_prompts(prompts)?;
        if table.dim()? != self.prompts.config.savpe.prompt_dim {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE SAVPE embedding dim {} does not match config dim {}",
                table.dim()?,
                self.prompts.config.savpe.prompt_dim
            )));
        }
        self.prompt_table = Some(table);
        Ok(())
    }

    /// Sets a prompt-free vocabulary without explicit prompt embeddings.
    pub fn set_prompt_free_vocabulary(&mut self, classes: Vec<String>) -> crate::Result<()> {
        self.prompts.set_prompt_free_vocabulary(classes)?;
        self.prompt_table = None;
        Ok(())
    }

    /// Sets a prompt-free vocabulary together with static LRPC embeddings.
    pub fn set_prompt_free_embeddings(&mut self, table: EmbeddingTable) -> crate::Result<()> {
        self.prompts
            .set_prompt_free_vocabulary(table.classes.clone())?;
        self.prompt_table = Some(table);
        Ok(())
    }

    /// Scores region features against the active prompt embedding table.
    pub fn score_region_features(&self, region_features: &Tensor) -> crate::Result<Tensor> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE prompt scoring requires precomputed prompt embeddings".to_string(),
            )
        })?;
        table.score_features(region_features, self.scorer)
    }

    /// Scores a dense feature map with the active prompt embedding table.
    pub fn score_feature_map(&self, feature_map: &Tensor) -> crate::Result<Tensor> {
        self.ensure_ready_for_validation()?;
        let table = self.prompt_table.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE feature-map scoring requires precomputed prompt embeddings".to_string(),
            )
        })?;
        Contrastive {
            scorer: self.scorer,
            ..Contrastive::default()
        }
        .forward(feature_map, table)
    }

    /// Encodes the active visual prompt masks into a prompt embedding table.
    ///
    /// `visuals` is the `[1, classes, h, w]` SAVPE-input tensor (build it with
    /// [`Visuals::from_boxes`](crate::yoloe::visuals::Visuals::from_boxes) /
    /// [`Visuals::from_masks`](crate::yoloe::visuals::Visuals::from_masks)). It is
    /// re-merged against the session's active prompts so already-merged
    /// `visuals` pass through unchanged.
    pub fn encode_visuals(
        &mut self,
        embedding_map: &Tensor,
        visuals: &crate::yoloe::visuals::Visuals,
    ) -> crate::Result<()> {
        if self.prompts.active_usage() != Some(Usage::Visual) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt mask encoding requires active visual prompts".to_string(),
            ));
        }
        let classes = self.prompts.active_classes().ok_or_else(|| {
            crate::Error::InvalidConfig("YOLOE visual prompts are empty".to_string())
        })?;
        let merged = self.merged_visual_prompt_masks(&visuals.tensor)?;
        let table = Pooler::default().encode_single_image_table(embedding_map, &merged, classes)?;
        self.prompt_table = Some(table);
        Ok(())
    }
}
