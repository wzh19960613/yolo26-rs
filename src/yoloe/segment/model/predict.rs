use candle_core::Tensor;

use crate::yoloe::head::lrpc::output::OfficialSegmentParts;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::segment::head::HeadParts;
use crate::yoloe::segment::model::Model;
use crate::yoloe::usage::Usage;

impl Model {
    /// Runs the full network and returns raw open-vocabulary segment head parts.
    pub fn forward_parts(&self, input: &Tensor, session: &Session) -> crate::Result<HeadParts> {
        self.network.forward_parts(input, session)
    }

    /// Runs the full network and returns top-k predictions plus prototype masks.
    pub fn forward_tensor(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<(Tensor, Tensor)> {
        self.network.forward(input, session)
    }

    /// Runs the full network through official prompt-free LRPC and returns raw selected parts.
    pub fn forward_prompt_free_parts(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<OfficialSegmentParts> {
        self.network.forward_prompt_free_parts(input, session)
    }

    /// Runs the full network through official prompt-free LRPC and returns top-k predictions.
    pub fn forward_prompt_free_tensor(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<(Tensor, Tensor)> {
        if session.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt-free tensor forward requires active prompt-free vocabulary"
                    .to_string(),
            ));
        }
        self.network.forward(input, session)
    }

    /// Runs the active prompt task for one source image and returns typed
    /// segmentation predictions (boxes + masks in source-image coordinates).
    ///
    /// This mirrors [`crate::segment::Model::predict`], decoding the
    /// open-vocabulary `(predictions, proto)` output through the shared
    /// segment postprocessor. Use this for text- or prompt-free inference.
    pub fn predict(
        &self,
        image: &crate::Image,
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
        let (preds, proto) = self.forward_tensor(&input, session)?;
        crate::segment::postprocess_segmentation(
            &preds,
            &proto,
            &letterbox_info,
            (image.width, image.height),
            filter,
            mask,
        )
    }

    /// Runs the prompt-free LRPC task for one source image and returns typed
    /// segmentation predictions.
    ///
    /// The session must have an active prompt-free vocabulary.
    pub fn predict_prompt_free(
        &self,
        image: &crate::Image,
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
        let (preds, proto) = self.forward_prompt_free_tensor(&input, session)?;
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
