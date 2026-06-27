use candle_core::Tensor;

use crate::yoloe::detect::model::Model;
use crate::yoloe::detect::model::network::Network;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::prompt::visual::Visual;
use crate::yoloe::usage::Usage;

impl Model {
    /// Runs online visual-prompt detection from source-image box or mask prompts.
    ///
    /// Prompt boxes/masks are expressed in source-image coordinates. The method
    /// applies the same rect letterbox used for YOLOE inference, encodes the
    /// prompt masks with the checkpoint's official SAVPE module, and decodes
    /// detection predictions back into source-image coordinates.
    ///
    /// This is the **intra-image** path: prompts and recognition happen on the
    /// same image. To reuse visual prompts across images, call
    /// [`Model::encode_visual_prompts`](super::Model::encode_visual_prompts)
    /// on a reference image and feed the returned table to
    /// [`Session::text_with_embeddings`](super::Session::text_with_embeddings).
    ///
    /// `source` selects whether prompts come from their stored xyxy boxes
    /// ([`super::VisualSource::Boxes`]) or from a source-image mask tensor
    /// ([`super::VisualSource::Masks`]).
    ///
    /// Detection-only checkpoints (`yoloe-26s.pt`) ship without SAVPE weights,
    /// so this path requires a checkpoint that carries them — typically a
    /// segmentation checkpoint (`*-seg.pt`) loaded through the detect head, or
    /// any YOLOE checkpoint whose `*.savpe.*` key family is present.
    pub fn predict_visual_prompts(
        &self,
        image: &crate::Image,
        prompts: &[Visual],
        source: crate::yoloe::visuals::VisualSource<'_>,
        session: &Session,
        filter: &crate::FilterOption,
    ) -> crate::Result<Vec<crate::detect::Prediction>> {
        let (input, letterbox_info) = crate::model::letterbox_rect(
            image,
            self.image_size,
            32,
            self.dtype,
            &self.config.device,
        )?;
        let preds = self.network.forward_visual_prompts(
            &input,
            &letterbox_info,
            prompts,
            source,
            session,
        )?;
        crate::detect::postprocess(&preds, &letterbox_info, image.width, image.height, filter)
    }
}

impl Network {
    /// Intra-image visual-prompt forward: SAVPE-encode the prompt masks on the
    /// same image, seed the session, and run the open-vocabulary detect head.
    pub(crate) fn forward_visual_prompts(
        &self,
        input: &Tensor,
        letterbox: &crate::model::LetterboxInfo,
        prompts: &[Visual],
        source: crate::yoloe::visuals::VisualSource<'_>,
        session: &Session,
    ) -> crate::Result<Tensor> {
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
        visual_session.forward_open_vocabulary_head(self.prompt_head()?, &head_features)
    }
}
