use candle_core::Tensor;

use crate::yoloe::detect::head::HeadParts;
use crate::yoloe::detect::model::Model;
use crate::yoloe::head::lrpc::output::OfficialPyramidOutput;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::usage::Usage;

impl Model {
    /// Runs the full network and returns raw open-vocabulary detection head parts.
    pub fn forward_parts(&self, input: &Tensor, session: &Session) -> crate::Result<HeadParts> {
        self.network.forward_parts(input, session)
    }

    /// Runs the full network and returns top-k detection predictions `[batch, det, 6]`.
    pub fn forward_tensor(&self, input: &Tensor, session: &Session) -> crate::Result<Tensor> {
        self.network.forward(input, session)
    }

    /// Runs the full network through official prompt-free LRPC and returns raw selected parts.
    pub fn forward_prompt_free_parts(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<OfficialPyramidOutput> {
        self.network.forward_prompt_free_parts(input, session)
    }

    /// Runs the full network through official prompt-free LRPC and returns top-k predictions.
    pub fn forward_prompt_free_tensor(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<Tensor> {
        if session.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt-free tensor forward requires active prompt-free vocabulary"
                    .to_string(),
            ));
        }
        self.network.forward(input, session)
    }

    /// Runs the active text or prompt-free prompt task for one source image
    /// and returns typed detection predictions in source-image coordinates.
    pub fn predict(
        &self,
        image: &crate::Image,
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
        let preds = self.forward_tensor(&input, session)?;
        crate::detect::postprocess(&preds, &letterbox_info, image.width, image.height, filter)
    }

    /// Runs the prompt-free LRPC task for one source image and returns typed
    /// detection predictions.
    ///
    /// The session must have an active prompt-free vocabulary.
    pub fn predict_prompt_free(
        &self,
        image: &crate::Image,
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
        let preds = self.forward_prompt_free_tensor(&input, session)?;
        crate::detect::postprocess(&preds, &letterbox_info, image.width, image.height, filter)
    }
}
