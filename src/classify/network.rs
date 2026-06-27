use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::network::backbone;

use super::{Config, head};

pub(crate) struct Network {
    backbone: backbone::Classify,
    head: head::Head,
}

impl Network {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let backbone = backbone::Classify::load(vb.clone(), config.scale)?;
        let input_channels = config.scale.channel(1024);
        let head = head::Head::load(vb.pp("10"), input_channels, config.labels_count)?;
        Ok(Self { backbone, head })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        self.head.forward(&features)
    }

    #[cfg(feature = "train")]
    pub(crate) fn forward_logits(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        self.head.forward_logits(&features)
    }
}
