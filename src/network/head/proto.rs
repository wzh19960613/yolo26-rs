use candle_core::{Module, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2dConfig, VarBuilder, conv_transpose2d, conv2d,
};

use crate::network::blocks::{ConvBlock, pytorch_conv2d};

pub struct Proto26 {
    refine: Vec<ConvBlock>,
    fuse: ConvBlock,
    cv1: ConvBlock,
    upsample: candle_nn::ConvTranspose2d,
    cv2: ConvBlock,
    cv3: ConvBlock,
    semantic: Option<SemanticBranch>,
}

pub struct Proto26Output {
    pub proto: Tensor,
    pub semantic: Option<Tensor>,
}

impl Proto26 {
    pub fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        proto_channels: usize,
        mask_channels: usize,
    ) -> Result<Self> {
        Self::load_inner(vb, input_channels, proto_channels, mask_channels, None)
    }

    pub fn load_with_semantic(
        vb: VarBuilder,
        input_channels: &[usize],
        proto_channels: usize,
        mask_channels: usize,
        semantic_classes: usize,
    ) -> Result<Self> {
        Self::load_inner(
            vb,
            input_channels,
            proto_channels,
            mask_channels,
            Some(semantic_classes),
        )
    }

    fn load_inner(
        vb: VarBuilder,
        input_channels: &[usize],
        proto_channels: usize,
        mask_channels: usize,
        semantic_classes: Option<usize>,
    ) -> Result<Self> {
        let mut refine = Vec::with_capacity(input_channels.len().saturating_sub(1));
        for (i, &channels) in input_channels.iter().skip(1).enumerate() {
            refine.push(ConvBlock::load(
                vb.pp("feat_refine").pp(i.to_string()),
                channels,
                input_channels[0],
                1,
                1,
                1,
                true,
            )?);
        }

        let cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };

        Ok(Self {
            refine,
            fuse: ConvBlock::load(
                vb.pp("feat_fuse"),
                input_channels[0],
                proto_channels,
                3,
                1,
                1,
                true,
            )?,
            cv1: ConvBlock::load(vb.pp("cv1"), proto_channels, proto_channels, 3, 1, 1, true)?,
            upsample: conv_transpose2d(proto_channels, proto_channels, 2, cfg, vb.pp("upsample"))?,
            cv2: ConvBlock::load(vb.pp("cv2"), proto_channels, proto_channels, 3, 1, 1, true)?,
            cv3: ConvBlock::load(vb.pp("cv3"), proto_channels, mask_channels, 1, 1, 1, true)?,
            semantic: semantic_classes
                .map(|classes| {
                    SemanticBranch::load(
                        vb.pp("semseg"),
                        input_channels[0],
                        proto_channels,
                        classes,
                    )
                })
                .transpose()?,
        })
    }

    pub fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        Ok(self.forward_inner(features, false)?.proto)
    }

    #[cfg(feature = "train")]
    pub fn forward_training(&self, features: &[&Tensor]) -> Result<Proto26Output> {
        self.forward_inner(features, true)
    }

    fn forward_inner(&self, features: &[&Tensor], include_semantic: bool) -> Result<Proto26Output> {
        let mut feat = features[0].clone();
        let (_, _, h, w) = feat.dims4()?;

        for (i, refine) in self.refine.iter().enumerate() {
            let up = refine.forward(features[i + 1])?.upsample_nearest2d(h, w)?;
            feat = feat.broadcast_add(&up)?;
        }

        let x = self.fuse.forward(&feat)?;
        let x = self.cv1.forward(&x)?;
        let x = self.upsample.forward(&x)?;
        let x = self.cv2.forward(&x)?;
        let proto = self.cv3.forward(&x)?;
        let semantic = if include_semantic {
            self.semantic
                .as_ref()
                .map(|semantic| semantic.forward(&feat))
                .transpose()?
        } else {
            None
        };
        Ok(Proto26Output { proto, semantic })
    }
}

struct SemanticBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    cv2: Conv2d,
}

impl SemanticBranch {
    fn load(vb: VarBuilder, input_channels: usize, hidden: usize, classes: usize) -> Result<Self> {
        let cfg = Conv2dConfig::default();
        Ok(Self {
            cv0: ConvBlock::load(vb.pp("0"), input_channels, hidden, 3, 1, 1, true)?,
            cv1: ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1, 1, true)?,
            cv2: pytorch_conv2d(hidden, classes, 1, cfg, vb.pp("2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.cv0.forward(x)?;
        let x = self.cv1.forward(&x)?;
        self.cv2.forward(&x)
    }
}
