use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::{box_branch::BoxBranch, cls_branch::ClsBranch};

use crate::yoloe::detect::head::Head;
use crate::yoloe::head::contrastive::{BnContrastive, Contrastive};
use crate::yoloe::select_lrpc_indices::has_bn_contrastive_tensors;

#[derive(Clone, Copy)]
struct LoadSpec {
    embed_dim: usize,
    max_det: usize,
    contrastive: Contrastive,
    use_bn_contrastive: bool,
    cls_hidden: Option<usize>,
    box_hidden: Option<usize>,
}

impl Head {
    /// Loads a YOLOE open-vocabulary head from official-style head weights.
    pub fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        contrastive: Contrastive,
    ) -> crate::Result<Self> {
        Self::load_with_hidden(
            vb,
            input_channels,
            embed_dim,
            max_det,
            contrastive,
            None,
            None,
        )
    }

    /// Loads a YOLOE head with explicit checkpoint-inferred hidden widths, used
    /// to match the official layout. `None` falls back to the formula defaults.
    pub fn load_with_hidden(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        contrastive: Contrastive,
        cls_hidden: Option<usize>,
        box_hidden: Option<usize>,
    ) -> crate::Result<Self> {
        let use_bn_contrastive = has_bn_contrastive_tensors(&vb);
        Self::load_inner(
            vb,
            input_channels,
            LoadSpec {
                embed_dim,
                max_det,
                contrastive,
                use_bn_contrastive,
                cls_hidden,
                box_hidden,
            },
        )
    }

    /// Loads a YOLOE head and requires official `BNContrastiveHead` weights.
    pub fn load_with_bn_contrastive(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        contrastive: Contrastive,
    ) -> crate::Result<Self> {
        Self::load_with_bn_contrastive_hidden(
            vb,
            input_channels,
            embed_dim,
            max_det,
            contrastive,
            None,
            None,
        )
    }

    /// Loads a YOLOE head requiring official `BNContrastiveHead` weights, with
    /// explicit checkpoint-inferred hidden widths.
    pub fn load_with_bn_contrastive_hidden(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        contrastive: Contrastive,
        cls_hidden: Option<usize>,
        box_hidden: Option<usize>,
    ) -> crate::Result<Self> {
        Self::load_inner(
            vb,
            input_channels,
            LoadSpec {
                embed_dim,
                max_det,
                contrastive,
                use_bn_contrastive: true,
                cls_hidden,
                box_hidden,
            },
        )
    }

    fn load_inner(vb: VarBuilder, input_channels: &[usize], spec: LoadSpec) -> crate::Result<Self> {
        if spec.embed_dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE open-vocabulary head embed_dim must be greater than zero".to_string(),
            ));
        }
        let reg_max = 1;
        // Hidden widths default to the historical formulas, but official
        // checkpoints use fixed sizes that differ from the formulas (e.g. Scale::N
        // cls hidden is 80, not 100). When the caller supplies checkpoint-inferred
        // widths, prefer those to stay aligned with the official layout.
        let c2 = spec
            .box_hidden
            .unwrap_or_else(|| 16_usize.max(input_channels[0] / 4).max(reg_max * 4));
        let c3 = spec
            .cls_hidden
            .unwrap_or_else(|| input_channels[0].max(spec.embed_dim.min(100)));
        let cfg = Conv2dConfig::default();
        let mut box_branches = Vec::with_capacity(input_channels.len());
        let mut embedding_branches = Vec::with_capacity(input_channels.len());
        for (i, &channels) in input_channels.iter().enumerate() {
            box_branches.push(BoxBranch::load(
                vb.pp("one2one_cv2").pp(i.to_string()),
                channels,
                c2,
                reg_max,
                cfg,
            )?);
            embedding_branches.push(ClsBranch::load(
                vb.pp("one2one_cv3").pp(i.to_string()),
                channels,
                c3,
                spec.embed_dim,
                cfg,
            )?);
        }
        let bn_contrastive_heads = if spec.use_bn_contrastive {
            let mut heads = Vec::with_capacity(input_channels.len());
            for i in 0..input_channels.len() {
                heads.push(BnContrastive::load(
                    vb.pp("one2one_cv4").pp(i.to_string()),
                    spec.embed_dim,
                )?);
            }
            Some(heads)
        } else {
            None
        };
        Ok(Self {
            box_branches,
            embedding_branches,
            bn_contrastive_heads,
            strides: [8.0, 16.0, 32.0],
            embed_dim: spec.embed_dim,
            max_det: spec.max_det,
            contrastive: spec.contrastive,
        })
    }

    /// Prompt embedding dimension used by this head.
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Maximum number of predictions kept by top-k postprocessing.
    pub const fn max_det(&self) -> usize {
        self.max_det
    }
}
