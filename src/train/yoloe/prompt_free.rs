//! Prompt-free training forward for the trainable YOLOE segmentation model.

use candle_core::Tensor;

use super::head::TrainableSegHead;
use super::model::Model;
use super::output::Output;

impl Model {
    /// Runs the official prompt-free `-seg-pf` head in dense training form.
    ///
    /// This path requires no text or visual prompt embeddings. It reuses the
    /// 2-layer box/embed stems and the 4585-class LRPC vocabulary head, matching
    /// the official `yoloe-26*-seg-pf.pt` layout.
    pub fn forward_prompt_free_dense(&self, input: &Tensor) -> crate::Result<Output> {
        let head = match &self.head {
            TrainableSegHead::PromptFree(head) => head,
            TrainableSegHead::Prompted(_) => {
                return Err(crate::Error::InvalidConfig(
                    "YOLOE prompt-free forward requires a PromptFree model".to_string(),
                ));
            }
        };
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        head.forward_train(&head_features)
    }
}
