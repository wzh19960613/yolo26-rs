use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::yoloe::prompt::session::Session;
use crate::yoloe::prompt::visual::Visual;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::segment::model::Model;
use crate::yoloe::segment::model::config::Config;
use crate::yoloe::segment::model::network::Network;
use crate::yoloe::usage::Usage;

impl Model {
    /// Runs online visual-prompt segmentation from source-image box or mask prompts.
    ///
    /// Prompt boxes/masks are expressed in source-image coordinates. The method
    /// applies the same rect letterbox used for YOLOE inference, encodes the
    /// prompt masks with the checkpoint's official SAVPE module, and decodes
    /// segmentation predictions back into source-image coordinates.
    ///
    /// This is the **intra-image** path: prompts and recognition happen on the
    /// same image. To reuse visual prompts across images, call
    /// [`Model::encode_visual_prompts`](super::Model::encode_visual_prompts)
    /// on a reference image and feed the returned table to
    /// [`Session::text_with_embeddings`](super::Session::text_with_embeddings).
    ///
    /// `source` selects whether prompts come from their stored xyxy boxes
    /// ([`super::VisualSource::Boxes`]) or from a source-image mask tensor
    /// ([`super::VisualSource::Masks`]) shaped `[prompts, H, W]` or
    /// `[1, prompts, H, W]`.
    pub fn predict_visual_prompts(
        &self,
        image: &crate::Image,
        prompts: &[Visual],
        source: crate::yoloe::visuals::VisualSource<'_>,
        session: &Session,
        filter: &crate::FilterOption,
        mask: &crate::MaskOption,
    ) -> crate::Result<Vec<crate::segment::Prediction>> {
        let (input, letterbox_info) = crate::model::letterbox_rect(
            image,
            self.image_size,
            32,
            self.dtype,
            &self.config.device,
        )?;
        let (preds, proto) = self.network.forward_visual_prompts(
            &input,
            &letterbox_info,
            prompts,
            source,
            session,
        )?;
        crate::segment::postprocess_segmentation(
            &preds,
            &proto,
            &letterbox_info,
            (image.width, image.height),
            filter,
            mask,
        )
    }
}

impl Network {
    pub(crate) fn load_savpe(
        vb: VarBuilder,
        config: &Config,
        input_channels: &[usize],
    ) -> crate::Result<Option<Encoder>> {
        crate::yoloe::savpe::encoder_load::load_savpe_gated(
            vb,
            config.official_savpe,
            config.prompt_head,
            config.savpe_hidden,
            config.embed_dim,
            input_channels,
        )
    }

    /// Intra-image visual-prompt forward: SAVPE-encode the prompt masks on the
    /// same image, seed the session, and run the open-vocabulary segment head.
    pub(crate) fn forward_visual_prompts(
        &self,
        input: &Tensor,
        letterbox: &crate::model::LetterboxInfo,
        prompts: &[Visual],
        source: crate::yoloe::visuals::VisualSource<'_>,
        session: &Session,
    ) -> crate::Result<(Tensor, Tensor)> {
        let encoded = self.encode_visual_prompt_table(input, letterbox, prompts, source)?;
        let mut visual_session = session.clone();
        visual_session.set_visual_prompt_embeddings(prompts.to_vec(), encoded.table)?;
        if visual_session.prompts.active_usage() != Some(Usage::Visual) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual-prompt inference did not activate visual prompts".to_string(),
            ));
        }
        let head = &encoded.head_features;
        let head_features = [&head[0], &head[1], &head[2]];
        visual_session.forward_open_vocabulary_segment_head(self.prompt_head()?, &head_features)
    }
}
