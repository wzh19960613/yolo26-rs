//! Cross-image visual-prompt encoding for the YOLOE detect-only model.
//!
//! [`Model::encode_visual_prompts`] runs the reference-image
//! letterbox + backbone + neck + SAVPE pipeline and returns an image-agnostic
//! [`EmbeddingTable`] (the official `vpe`). The shared tail
//! [`Network::encode_visual_prompt_table`] is also reused
//! by the intra-image `forward_visual_prompts` path to avoid recomputing the
//! backbone/neck features.

use candle_core::Tensor;

use crate::yoloe::detect::model::Model;
use crate::yoloe::detect::model::network::Network;
use crate::yoloe::prompt::visual::{Visual, visual_prompt_classes};
use crate::yoloe::prompt::visual_letterbox::{box_masks_for_letterbox, mask_prompts_for_letterbox};
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::{VisualSource, Visuals};

/// Returned shared backbone/neck head features for one image.
pub(crate) struct EncodedPrompt {
    /// Image-agnostic prompt embedding table (official `vpe`).
    pub table: EmbeddingTable,
    /// Small / medium / large head features the table was encoded from.
    pub head_features: [Tensor; 3],
}

impl Model {
    /// Encodes visual prompts on a reference image into a reusable,
    /// image-agnostic prompt embedding table (the official `vpe`).
    ///
    /// This is the two-step counterpart of intra-image
    /// [`predict_visual_prompts`](super::Model::predict_visual_prompts):
    /// it runs the reference image through the backbone/neck and the official
    /// SAVPE module to produce a `[classes, embed_dim]` table that is decoupled
    /// from the reference image. Feed the returned table to
    /// [`Session::text_with_embeddings`](super::Session::text_with_embeddings)
    /// to detect/recall the same classes on **any** other image via the normal
    /// [`predict`](super::Model::predict) path.
    ///
    /// Prompt boxes/masks are expressed in `reference_image` coordinates.
    /// `source` selects whether prompts come from their stored xyxy boxes
    /// ([`VisualSource::Boxes`]) or from a source-image mask tensor
    /// ([`VisualSource::Masks`]). Detection-only checkpoints
    /// (`yoloe-26s.pt`) ship without SAVPE weights, so this path requires a
    /// checkpoint that carries them — typically a `-seg` checkpoint.
    pub fn encode_visual_prompts(
        &self,
        reference_image: &crate::Image,
        prompts: &[Visual],
        source: VisualSource<'_>,
    ) -> crate::Result<EmbeddingTable> {
        let encoded = self.encode_visual_prompt_table(reference_image, prompts, source)?;
        Ok(encoded.table)
    }

    pub(crate) fn encode_visual_prompt_table(
        &self,
        reference_image: &crate::Image,
        prompts: &[Visual],
        source: VisualSource<'_>,
    ) -> crate::Result<EncodedPrompt> {
        let (input, letterbox_info) = crate::model::letterbox_rect(
            reference_image,
            self.image_size,
            32,
            self.dtype,
            &self.config.device,
        )?;
        self.network
            .encode_visual_prompt_table(&input, &letterbox_info, prompts, source)
    }
}

impl Network {
    /// Backbone + neck + SAVPE encoding tail shared by the cross-image and
    /// intra-image visual-prompt paths. Returns the image-agnostic prompt table
    /// plus the head features it was encoded from (so the intra-image path can
    /// reuse them for the head forward without recomputation).
    pub(crate) fn encode_visual_prompt_table(
        &self,
        input: &Tensor,
        letterbox: &crate::model::LetterboxInfo,
        prompts: &[Visual],
        source: VisualSource<'_>,
    ) -> crate::Result<EncodedPrompt> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        let (_, _, feature_h, feature_w) = head_features[0].dims4()?;
        let prompt_masks =
            build_prompt_masks(prompts, source, letterbox, feature_h, feature_w, input)?;
        let savpe = require_savpe(self.savpe.as_ref())?;
        let table = savpe.encode_single_image_table(
            &head_features,
            &prompt_masks,
            visual_prompt_classes(prompts),
        )?;
        Ok(EncodedPrompt {
            table,
            head_features: [pyramid.small, pyramid.medium, pyramid.large],
        })
    }
}

/// Dispatches box- vs. mask-prompt rasterization into SAVPE-input [`Visuals`].
pub(crate) fn build_prompt_masks(
    prompts: &[Visual],
    source: VisualSource<'_>,
    letterbox: &crate::model::LetterboxInfo,
    feature_h: usize,
    feature_w: usize,
    input: &Tensor,
) -> crate::Result<Visuals> {
    match source {
        VisualSource::Boxes => {
            box_masks_for_letterbox(prompts, letterbox, feature_h, feature_w, input)
        }
        VisualSource::Masks(source_masks) => {
            mask_prompts_for_letterbox(prompts, source_masks, letterbox, feature_h, feature_w)
        }
    }
}

/// Returns the SAVPE encoder or an error explaining that the checkpoint omits it.
pub(crate) fn require_savpe(savpe: Option<&Encoder>) -> crate::Result<&Encoder> {
    savpe.ok_or_else(|| {
        crate::Error::InvalidConfig(
            "YOLOE detect visual-prompt inference requires official SAVPE weights; \
             this checkpoint does not carry the `*.savpe.*` key family \
             (detection-only checkpoints omit SAVPE — use a `-seg` checkpoint)"
                .to_string(),
        )
    })
}
