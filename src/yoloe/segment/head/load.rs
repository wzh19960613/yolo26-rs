use candle_nn::VarBuilder;

use crate::yoloe::head::contrastive::Contrastive;
use crate::yoloe::segment::head::Head;
use crate::yoloe::segment::head::build::LoadInnerSpec;

impl Head {
    /// Loads a YOLOE segmentation head from official-style head weights.
    ///
    /// The mask coefficient branch uses `one2one_cv5`, matching official
    /// `YOLOESegment` naming where `cv4` is reserved for contrastive heads.
    pub fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
        contrastive: Contrastive,
    ) -> crate::Result<Self> {
        Self::load_with_mask_branch(
            vb,
            input_channels,
            embed_dim,
            max_det,
            nm,
            npr,
            contrastive,
            "one2one_cv5",
        )
    }

    /// Loads a YOLOE segmentation head and requires official `BNContrastiveHead` weights.
    pub fn load_with_bn_contrastive(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
        contrastive: Contrastive,
    ) -> crate::Result<Self> {
        Self::load_with_hidden(
            vb,
            input_channels,
            embed_dim,
            max_det,
            nm,
            npr,
            contrastive,
            "one2one_cv5",
            None,
            None,
            None,
            true,
        )
    }

    /// Loads a YOLOE segmentation head with an explicit mask coefficient branch prefix.
    #[expect(
        clippy::too_many_arguments,
        reason = "public loader preserves the existing explicit head construction API"
    )]
    pub fn load_with_mask_branch(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
        contrastive: Contrastive,
        mask_branch: &str,
    ) -> crate::Result<Self> {
        Self::load_with_hidden(
            vb,
            input_channels,
            embed_dim,
            max_det,
            nm,
            npr,
            contrastive,
            mask_branch,
            None,
            None,
            None,
            false,
        )
    }

    /// Loads a YOLOE segmentation head with checkpoint-inferred hidden widths so
    /// the built branches match the official layout exactly.
    #[expect(
        clippy::too_many_arguments,
        reason = "public loader preserves the existing explicit head construction API"
    )]
    pub fn load_with_hidden(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
        contrastive: Contrastive,
        mask_branch: &str,
        cls_hidden: Option<usize>,
        box_hidden: Option<usize>,
        mask_hidden: Option<usize>,
        require_bn_contrastive: bool,
    ) -> crate::Result<Self> {
        Self::load_full(
            vb,
            input_channels,
            embed_dim,
            max_det,
            nm,
            npr,
            contrastive,
            mask_branch,
            cls_hidden,
            box_hidden,
            mask_hidden,
            require_bn_contrastive,
            false,
        )
    }

    /// Full head loader with explicit control over the one-to-many branch set.
    /// When `build_one_to_many` is true, also builds `cv2`/`cv3`/`cv4`/`cv5`
    /// (one-to-many) so the saved checkpoint matches the official symmetric
    /// `yoloe-26*-seg.pt` layout.
    #[expect(
        clippy::too_many_arguments,
        reason = "public loader preserves the existing explicit head construction API"
    )]
    pub fn load_full(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        max_det: usize,
        nm: usize,
        npr: usize,
        contrastive: Contrastive,
        mask_branch: &str,
        cls_hidden: Option<usize>,
        box_hidden: Option<usize>,
        mask_hidden: Option<usize>,
        require_bn_contrastive: bool,
        build_one_to_many: bool,
    ) -> crate::Result<Self> {
        Self::load_inner(
            vb,
            input_channels,
            LoadInnerSpec {
                embed_dim,
                max_det,
                nm,
                npr,
                contrastive,
                mask_branch,
                cls_hidden,
                box_hidden,
                mask_hidden,
                require_bn_contrastive,
                build_one_to_many,
            },
        )
    }

    /// Prompt embedding dimension used by the detection branch.
    pub const fn embed_dim(&self) -> usize {
        self.detect.embed_dim()
    }

    /// Number of mask coefficient channels.
    pub const fn mask_channels(&self) -> usize {
        self.nm
    }
}
