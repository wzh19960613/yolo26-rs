use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Init, VarBuilder, conv2d};

use crate::network::blocks::{ConvBlock, pytorch_conv2d_init};

pub struct ClsBranch {
    dw0: ConvBlock,
    cv0: ConvBlock,
    dw1: ConvBlock,
    cv1: ConvBlock,
    cv2: candle_nn::Conv2d,
    out_channels: usize,
}

impl ClsBranch {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        hidden: usize,
        nc: usize,
        cfg: Conv2dConfig,
    ) -> Result<Self> {
        Self::load_inner(vb, channels, hidden, nc, cfg, None)
    }

    pub(crate) fn load_with_class_bias(
        vb: VarBuilder,
        channels: usize,
        hidden: usize,
        nc: usize,
        cfg: Conv2dConfig,
        stride: f64,
    ) -> Result<Self> {
        Self::load_inner(vb, channels, hidden, nc, cfg, Some(stride))
    }

    fn load_inner(
        vb: VarBuilder,
        channels: usize,
        hidden: usize,
        nc: usize,
        cfg: Conv2dConfig,
        class_stride: Option<f64>,
    ) -> Result<Self> {
        Ok(Self {
            dw0: ConvBlock::load(vb.pp("0").pp("0"), channels, channels, 3, 1, channels, true)?,
            cv0: ConvBlock::load(vb.pp("0").pp("1"), channels, hidden, 1, 1, 1, true)?,
            dw1: ConvBlock::load(vb.pp("1").pp("0"), hidden, hidden, 3, 1, hidden, true)?,
            cv1: ConvBlock::load(vb.pp("1").pp("1"), hidden, hidden, 1, 1, 1, true)?,
            cv2: load_projection(vb.pp("2"), hidden, nc, cfg, class_stride)?,
            out_channels: nc,
        })
    }

    pub fn forward_map(&self, feature: &Tensor) -> Result<Tensor> {
        self.cv2.forward(&self.pre_projection_map(feature)?)
    }

    pub(crate) fn pre_projection_map(&self, feature: &Tensor) -> Result<Tensor> {
        let cx = self.dw0.forward(feature)?;
        let cx = self.cv0.forward(&cx)?;
        let cx = self.dw1.forward(&cx)?;
        self.cv1.forward(&cx)
    }

    pub fn forward(
        &self,
        feature: &Tensor,
        batch: usize,
        nc: usize,
        spatial: usize,
    ) -> Result<Tensor> {
        debug_assert_eq!(nc, self.out_channels);
        self.forward_map(feature)?.reshape((batch, nc, spatial))
    }
}

fn load_projection(
    vb: VarBuilder,
    hidden: usize,
    nc: usize,
    cfg: Conv2dConfig,
    class_stride: Option<f64>,
) -> Result<Conv2d> {
    let Some(stride) = class_stride else {
        return conv2d(hidden, nc, 1, cfg, vb);
    };
    let weight = vb.get_with_hints(
        (nc, hidden / cfg.groups, 1, 1),
        "weight",
        pytorch_conv2d_init(hidden, 1, cfg.groups),
    )?;
    let bias = vb.get_with_hints(nc, "bias", Init::Const(class_bias(nc, stride)))?;
    Ok(Conv2d::new(weight, Some(bias), cfg))
}

fn class_bias(nc: usize, stride: f64) -> f64 {
    let classes = nc.max(1) as f64;
    let cells = (640.0 / stride.max(f64::EPSILON)).powi(2);
    (5.0 / classes / cells).ln()
}
