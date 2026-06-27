use std::collections::BTreeSet;

/// One visual prompt region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Visual {
    /// Class id assigned to this visual exemplar.
    pub class_id: u32,
    /// Prompt bounds in source-image xyxy coordinates.
    pub xyxy: [f32; 4],
    /// Source annotation kind used to build the visual prompt mask.
    pub kind: VisualKind,
}

/// Source annotation kind for a YOLOE visual prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualKind {
    /// The prompt mask is generated from a source-image bounding box.
    Box,
    /// The prompt mask is supplied from a source-image segmentation mask.
    Mask,
}

impl Visual {
    /// Creates a validated box visual prompt.
    pub fn from_box(class_id: u32, xyxy: [f32; 4]) -> crate::Result<Self> {
        Self::with_kind(class_id, xyxy, VisualKind::Box)
    }

    /// Creates a validated mask visual prompt.
    ///
    /// The `xyxy` bounds describe the source-image extent of the mask. The mask
    /// pixels themselves are supplied to the visual-prompt predict entry that
    /// takes a `VisualSource::Masks` source.
    pub fn from_mask(class_id: u32, xyxy: [f32; 4]) -> crate::Result<Self> {
        Self::with_kind(class_id, xyxy, VisualKind::Mask)
    }

    fn with_kind(class_id: u32, xyxy: [f32; 4], kind: VisualKind) -> crate::Result<Self> {
        if !xyxy.iter().all(|value| value.is_finite()) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt bounds must be finite".to_string(),
            ));
        }
        if xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1] {
            return Err(crate::Error::InvalidConfig(
                "YOLOE visual prompt bounds must have positive width and height".to_string(),
            ));
        }
        Ok(Self {
            class_id,
            xyxy,
            kind,
        })
    }
}

pub(crate) fn visual_prompt_classes(prompts: &[Visual]) -> Vec<String> {
    visual_prompt_class_ids(prompts)
        .into_iter()
        .map(|class_id| format!("visual_class_{class_id}"))
        .collect()
}

pub(crate) fn visual_prompt_class_ids(prompts: &[Visual]) -> Vec<u32> {
    prompts
        .iter()
        .map(|prompt| prompt.class_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn validate_visual_prompt_source(
    prompts: &[Visual],
    kind: VisualKind,
) -> crate::Result<()> {
    if prompts.is_empty() {
        return Err(crate::Error::InvalidConfig(
            "YOLOE visual prompts require at least one exemplar".to_string(),
        ));
    }
    if prompts.iter().any(|prompt| prompt.kind != kind) {
        return Err(crate::Error::InvalidConfig(
            "YOLOE visual prompt kind does not match requested mask builder".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn scaled_mask_dim(source: usize, scale_factor: f32) -> crate::Result<usize> {
    if !scale_factor.is_finite() || scale_factor <= 0.0 {
        return Err(crate::Error::InvalidConfig(
            "YOLOE visual prompt scale_factor must be positive and finite".to_string(),
        ));
    }
    Ok(((source as f32) * scale_factor).floor().max(1.0) as usize)
}
