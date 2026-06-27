use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::network::{DetectionNetwork, NetworkHead};

use super::{Config, head};

pub(crate) type Network = DetectionNetwork<head::Head>;

pub(crate) fn load(vb: VarBuilder, config: &Config) -> Result<Network> {
    DetectionNetwork::load(vb, config.scale, "23", |vb, ch| {
        head::Head::load(vb, ch, config.labels_count, config.max_predictions)
    })
}

impl NetworkHead for head::Head {
    type Output = Tensor;

    fn forward_features(&self, features: &[&Tensor]) -> Result<Self::Output> {
        self.forward(features)
    }
}
