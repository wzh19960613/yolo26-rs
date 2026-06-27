use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::network::blocks::ConvBlock;

pub(crate) struct OfficialFeaturePyramid {
    cls_branches: Vec<ClsFeatureBranch>,
    loc_branches: Vec<LocFeatureBranch>,
}

impl OfficialFeaturePyramid {
    pub(crate) fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        cls_hidden: usize,
        loc_hidden: usize,
    ) -> crate::Result<Self> {
        let mut cls_branches = Vec::with_capacity(input_channels.len());
        let mut loc_branches = Vec::with_capacity(input_channels.len());
        for (i, &channels) in input_channels.iter().enumerate() {
            cls_branches.push(ClsFeatureBranch::load(
                vb.pp("one2one_cv3").pp(i.to_string()),
                channels,
                cls_hidden,
            )?);
            loc_branches.push(LocFeatureBranch::load(
                vb.pp("one2one_cv2").pp(i.to_string()),
                channels,
                loc_hidden,
            )?);
        }
        Ok(Self {
            cls_branches,
            loc_branches,
        })
    }

    pub(crate) fn forward(
        &self,
        features: &[&Tensor],
    ) -> crate::Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut cls = Vec::with_capacity(features.len());
        let mut loc = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            cls.push(self.cls_branches[i].forward(feature)?);
            loc.push(self.loc_branches[i].forward(feature)?);
        }
        Ok((cls, loc))
    }
}

struct LocFeatureBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
}

impl LocFeatureBranch {
    fn load(vb: VarBuilder, channels: usize, hidden: usize) -> crate::Result<Self> {
        Ok(Self {
            cv0: ConvBlock::load(vb.pp("0"), channels, hidden, 3, 1, 1, true)?,
            cv1: ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1, 1, true)?,
        })
    }

    fn forward(&self, feature: &Tensor) -> crate::Result<Tensor> {
        let x = self.cv0.forward(feature)?;
        Ok(self.cv1.forward(&x)?)
    }
}

struct ClsFeatureBranch {
    dw0: ConvBlock,
    cv0: ConvBlock,
    dw1: ConvBlock,
    cv1: ConvBlock,
}

impl ClsFeatureBranch {
    fn load(vb: VarBuilder, channels: usize, hidden: usize) -> crate::Result<Self> {
        Ok(Self {
            dw0: ConvBlock::load(vb.pp("0").pp("0"), channels, channels, 3, 1, channels, true)?,
            cv0: ConvBlock::load(vb.pp("0").pp("1"), channels, hidden, 1, 1, 1, true)?,
            dw1: ConvBlock::load(vb.pp("1").pp("0"), hidden, hidden, 3, 1, hidden, true)?,
            cv1: ConvBlock::load(vb.pp("1").pp("1"), hidden, hidden, 1, 1, 1, true)?,
        })
    }

    fn forward(&self, feature: &Tensor) -> crate::Result<Tensor> {
        let x = self.dw0.forward(feature)?;
        let x = self.cv0.forward(&x)?;
        let x = self.dw1.forward(&x)?;
        Ok(self.cv1.forward(&x)?)
    }
}
