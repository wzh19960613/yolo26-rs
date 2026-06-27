use std::path::Path;

use crate::yoloe::head::key_plan::KeyPlan;
use crate::yoloe::select_lrpc_indices::{
    contains_marker, has_official_bn_contrastive, has_official_lrpc, has_official_reprta,
    has_official_savpe, has_tensor_prefix, infer_yoloe_head_prefix,
};

/// Key layout detected from a YOLOE checkpoint or converted SafeTensors file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// Number of tensor entries inspected.
    pub tensor_count: usize,
    /// Detected model head prefix, usually `model.23`.
    pub head_prefix: String,
    /// Whether `one2one_cv2` box branches are present.
    pub has_one2one_cv2: bool,
    /// Whether `one2one_cv3` embedding branches are present.
    pub has_one2one_cv3: bool,
    /// Whether `one2one_cv4` branches are present.
    pub has_one2one_cv4: bool,
    /// Whether official YOLOE `BNContrastiveHead` parameters are present under `one2one_cv4`.
    pub has_bn_contrastive: bool,
    /// Whether official YOLOE segment `one2one_cv5` mask branches are present.
    pub has_one2one_cv5: bool,
    /// Whether prototype mask weights are present.
    pub has_proto: bool,
    /// Whether keys containing RepRTA markers are present.
    pub has_reprta: bool,
    /// Whether official RepRTA key families are present.
    pub has_official_reprta: bool,
    /// Whether keys containing SAVPE markers are present.
    pub has_savpe: bool,
    /// Whether official SAVPE key families are present.
    pub has_official_savpe: bool,
    /// Whether keys containing LRPC markers are present.
    pub has_lrpc: bool,
    /// Whether official LRPC key families are present.
    pub has_official_lrpc: bool,
}

impl Layout {
    /// Builds a layout report from tensor names.
    pub fn from_tensor_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        let names = names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>();
        let head_prefix = infer_yoloe_head_prefix(&names);
        let has_one2one_cv2 = has_tensor_prefix(&names, &format!("{head_prefix}.one2one_cv2."));
        let has_one2one_cv3 = has_tensor_prefix(&names, &format!("{head_prefix}.one2one_cv3."));
        let has_one2one_cv4 = has_tensor_prefix(&names, &format!("{head_prefix}.one2one_cv4."));
        let has_bn_contrastive = has_official_bn_contrastive(&names, &head_prefix);
        let has_one2one_cv5 = has_tensor_prefix(&names, &format!("{head_prefix}.one2one_cv5."));
        let has_proto = has_tensor_prefix(&names, &format!("{head_prefix}.proto."));
        let has_official_savpe = has_official_savpe(&names, &head_prefix);
        let has_official_reprta = has_official_reprta(&names, &head_prefix);
        let has_official_lrpc = has_official_lrpc(&names, &head_prefix);
        Self {
            tensor_count: names.len(),
            head_prefix,
            has_one2one_cv2,
            has_one2one_cv3,
            has_one2one_cv4,
            has_bn_contrastive,
            has_one2one_cv5,
            has_proto,
            has_reprta: names.iter().any(|name| contains_marker(name, "reprta")),
            has_official_reprta,
            has_savpe: names.iter().any(|name| contains_marker(name, "savpe")),
            has_official_savpe,
            has_lrpc: names.iter().any(|name| contains_marker(name, "lrpc")),
            has_official_lrpc,
        }
    }

    /// Builds a layout report from SafeTensors bytes without loading tensor data.
    pub fn from_safetensors_bytes(bytes: &[u8]) -> crate::Result<Self> {
        let safetensors = candle_core::safetensors::SliceSafetensors::new(bytes)?;
        Ok(Self::from_tensor_names(
            safetensors
                .tensors()
                .into_iter()
                .map(|(name, _)| name)
                .collect::<Vec<_>>(),
        ))
    }

    /// Builds a layout report from a SafeTensors file.
    pub fn from_safetensors_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::from_safetensors_bytes(&std::fs::read(path)?)
    }

    /// Builds a layout report from an official `.pt` checkpoint, mirroring
    /// [`Self::from_safetensors_file`].
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        let device = candle_core::Device::Cpu;
        let names = crate::pt_loader::load_pt_to_tensors(path, &device)?
            .into_keys()
            .collect::<Vec<_>>();
        Ok(Self::from_tensor_names(names))
    }

    /// Returns missing key families for open-vocabulary detection.
    pub fn missing_detect(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.has_one2one_cv2 {
            missing.push(format!("{}.one2one_cv2.*", self.head_prefix));
        }
        if !self.has_one2one_cv3 {
            missing.push(format!("{}.one2one_cv3.*", self.head_prefix));
        }
        missing
    }

    /// Returns missing key families for official YOLOE open-vocabulary detection.
    pub fn missing_official_detect(&self) -> Vec<String> {
        let mut missing = self.missing_detect();
        if !self.has_bn_contrastive {
            missing.push(format!(
                "{}.one2one_cv4.* BNContrastiveHead",
                self.head_prefix
            ));
        }
        missing
    }

    /// Returns missing key families for official YOLOE segmentation.
    pub fn missing_official_segment(&self) -> Vec<String> {
        let mut missing = self.missing_official_detect();
        if !self.has_one2one_cv5 {
            missing.push(format!("{}.one2one_cv5.*", self.head_prefix));
        }
        if !self.has_proto {
            missing.push(format!("{}.proto.*", self.head_prefix));
        }
        missing
    }

    /// Returns missing key families for compatible YOLOE segmentation loading.
    ///
    /// Official YOLOE uses `one2one_cv5` for mask coefficients. This compatible
    /// check allows `one2one_cv4` as a fallback for converted closed-set segment
    /// checkpoints so that migration/parity diagnostics can still run.
    pub fn missing_compatible_segment(&self) -> Vec<String> {
        let mut missing = self.missing_detect();
        if self.compatible_segment_mask_branch().is_none() {
            missing.push(format!(
                "{}.one2one_cv5.* or {}.one2one_cv4.*",
                self.head_prefix, self.head_prefix
            ));
        }
        if !self.has_proto {
            missing.push(format!("{}.proto.*", self.head_prefix));
        }
        missing
    }

    /// Returns true when open-vocabulary detection key families are present.
    pub fn can_load_detect(&self) -> bool {
        self.missing_detect().is_empty()
    }

    /// Returns true when official YOLOE detection key families are present.
    pub fn can_load_official_detect(&self) -> bool {
        self.missing_official_detect().is_empty()
    }

    /// Returns true when official YOLOE segmentation key families are present.
    pub fn can_load_official_segment(&self) -> bool {
        self.missing_official_segment().is_empty()
    }

    /// Returns true when compatible segmentation key families are present.
    pub fn can_load_compatible_segment(&self) -> bool {
        self.missing_compatible_segment().is_empty()
    }

    /// Returns the compatible mask coefficient branch name.
    pub fn compatible_segment_mask_branch(&self) -> Option<&'static str> {
        if self.has_one2one_cv5 {
            Some("one2one_cv5")
        } else if self.has_one2one_cv4 {
            Some("one2one_cv4")
        } else {
            None
        }
    }

    /// Creates a head key plan when compatible open-vocabulary segmentation can be loaded.
    pub fn compatible_segment_head_plan(&self) -> crate::Result<KeyPlan> {
        if !self.can_load_compatible_segment() {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE checkpoint is missing key families: {}",
                self.missing_compatible_segment().join(", ")
            )));
        }
        Ok(KeyPlan {
            head_prefix: self.head_prefix.clone(),
            box_branch: format!("{}.one2one_cv2", self.head_prefix),
            embedding_branch: format!("{}.one2one_cv3", self.head_prefix),
            contrastive_branch: self
                .has_bn_contrastive
                .then(|| format!("{}.one2one_cv4", self.head_prefix)),
            segment_mask_branch: self
                .compatible_segment_mask_branch()
                .map(|branch| format!("{}.{}", self.head_prefix, branch)),
            proto: Some(format!("{}.proto", self.head_prefix)),
            uses_official_bn_contrastive: self.has_bn_contrastive,
            uses_official_segment_mask_branch: self.has_one2one_cv5,
        })
    }
}
