//! CLIP text encoder for YOLOE text-prompt class names.
//!
//! Backed by the `mobileclip2-b-rs` crate, which loads the official
//! MobileCLIP2-b weights and BPE tokenizer and produces CLIP-aligned
//! `[classes, 512]` L2-normalized embeddings. The encoder is an owned value
//! constructed once and reused across many `Session::text` calls, so the CLIP
//! model and tokenizer are loaded only once regardless of how many text-prompt
//! sessions are built.

use candle_core::{DType, Tensor};

use crate::Result;

/// Reusable CLIP text encoder producing `[classes, embed_dim]` L2-normalized
/// embeddings.
///
/// Wraps a loaded [`mobileclip2_b_rs::Model`] plus its bound tokenizer. Construct
/// once via [`ClipTextEncoder::new`] (wrap pre-built model + tokenizer),
/// [`ClipTextEncoder::from_files`] (paths), or [`ClipTextEncoder::from_bytes`]
/// (in-memory), then pass `&self` to as many
/// [`crate::yoloe::Session::text`] calls as needed. Embeddings are cast to F32
/// so they can be moved and aligned to any YOLOE model dtype/device during
/// scoring.
pub struct ClipTextEncoder {
    model: mobileclip2_b_rs::Model,
    tokenizer: mobileclip2_b_rs::Tokenizer,
}

impl ClipTextEncoder {
    /// Wraps an already-loaded [`mobileclip2_b_rs::Model`] and
    /// [`mobileclip2_b_rs::Tokenizer`] into a reusable text encoder.
    ///
    /// Use this when you need full control over the MobileCLIP2 config (device,
    /// dtype, context length, ...) — construct the model and tokenizer
    /// yourself, then hand them to `new`. For the common file/bytes cases see
    /// [`Self::from_files`] and [`Self::from_bytes`].
    pub fn new(model: mobileclip2_b_rs::Model, tokenizer: mobileclip2_b_rs::Tokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Loads the CLIP text encoder and tokenizer from `weights_path` and
    /// `tokenizer_path` on CPU.
    pub fn from_files(
        model_path: impl AsRef<std::path::Path>,
        tokenizer_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let config = mobileclip2_b_rs::Config::default();
        let model = mobileclip2_b_rs::Model::from_file_with(model_path.as_ref(), config)?;
        let tokenizer = mobileclip2_b_rs::Tokenizer::from_file(
            tokenizer_path.as_ref(),
            &mobileclip2_b_rs::Config::default(),
        )?;
        Ok(Self::new(model, tokenizer))
    }

    /// Loads the CLIP text encoder and tokenizer from in-memory bytes, no
    /// filesystem access required.
    pub fn from_bytes(model: impl AsRef<[u8]>, tokenizer: impl AsRef<[u8]>) -> Result<Self> {
        let config = mobileclip2_b_rs::Config::default();
        let model =
            mobileclip2_b_rs::Model::from_safetensors(model.as_ref().to_vec(), config.clone())?;
        let tokenizer = mobileclip2_b_rs::Tokenizer::from_bytes(tokenizer.as_ref(), &config)?;
        Ok(Self::new(model, tokenizer))
    }

    /// Encodes a slice of class names into an `[classes, embed_dim]` F32 tensor.
    ///
    /// Names are tokenized as-is (no prompt template), matching the official
    /// Ultralytics `get_text_pe(names)` behavior. The result is L2-normalized
    /// by the underlying MobileCLIP2-b encoder and cast to F32 so it can be
    /// moved and aligned to any YOLOE model dtype/device during scoring.
    ///
    /// Each name may be borrowed (`&str`) or owned (`String`); see
    /// [`Self::embed_texts`] for the `Into<Cow<str>>` version.
    pub fn embed_classes(&self, classes: &[&str]) -> Result<Tensor> {
        let embeddings = self.model.encode_texts(&self.tokenizer, classes)?;
        embeddings.to_dtype(DType::F32).map_err(Into::into)
    }

    /// Encodes class names supplied as any `AsRef<str>` (e.g. `&str`,
    /// `String`, `&&str`) into an `[classes, embed_dim]` F32 tensor. Same
    /// semantics as [`Self::embed_classes`] but avoids forcing the caller to
    /// build a `Vec<&str>` manually.
    pub fn embed_texts<I, S>(&self, classes: I) -> Result<Tensor>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let owned: Vec<String> = classes.into_iter().map(|s| s.as_ref().to_owned()).collect();
        let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
        self.embed_classes(&refs)
    }
}
