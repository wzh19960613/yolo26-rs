use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::network::{backbone, neck};

use super::{Config, head};

pub(crate) struct Network {
    backbone: backbone::Base,
    neck: neck::Semantic,
    head: head::Head,
}

impl Network {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let backbone = backbone::Base::load(vb.clone(), config.scale)?;
        let neck = neck::Semantic::load(vb.clone(), config.scale)?;
        let input_channels = [config.scale.channel(256), config.scale.channel(512)];
        let head = head::Head::load(vb.pp("17"), &input_channels, config.labels_count)?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        self.head.forward(&[&pyramid.small, &pyramid.medium])
    }
}
