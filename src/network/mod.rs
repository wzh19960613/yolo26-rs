// Network internals include task-specific submodules (classify/semantic/pose/obb
// branches, Proto26, etc.) that are only exercised when the corresponding task
// feature is enabled. Silence dead-code/unused-import noise for those configs.
#![allow(dead_code, unused_imports)]

pub(crate) mod backbone;
pub(crate) mod blocks;
pub(crate) mod head;
pub(crate) mod neck;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::Scale;

pub(crate) trait NetworkHead {
    type Output;
    fn forward_features(&self, features: &[&Tensor]) -> Result<Self::Output>;
}

pub(crate) struct DetectionNetwork<H: NetworkHead> {
    backbone: backbone::Base,
    neck: neck::Base,
    pub(crate) head: H,
}

impl<H: NetworkHead> DetectionNetwork<H> {
    pub(crate) fn load(
        vb: VarBuilder,
        scale: Scale,
        head_path: &str,
        load_head: impl FnOnce(VarBuilder, &[usize]) -> Result<H>,
    ) -> Result<Self> {
        let backbone = backbone::Base::load(vb.clone(), scale)?;
        let neck = neck::Base::load(vb.clone(), scale)?;
        let input_channels = scale.head_input_channels();
        let head = load_head(vb.pp(head_path), &input_channels)?;
        Ok(Self {
            backbone,
            neck,
            head,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor) -> Result<H::Output> {
        let pyramid = self.forward_pyramid(input)?;
        let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        self.head.forward_features(&features)
    }

    pub(crate) fn forward_pyramid(&self, input: &Tensor) -> Result<neck::base::Output> {
        let features = self.backbone.forward(input)?;
        self.neck.forward(&features)
    }
}
