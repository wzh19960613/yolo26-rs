//! Per-task native config builders, expressed as `impl WasmConfig` blocks.
//!
//! Each task compiles its own builder only when the matching feature is on, so
//! the WASM bindings track the feature matrix without `cfg` noise in the main
//! [`WasmConfig`](super::config::WasmConfig) definition.

use super::config::WasmConfig;

#[cfg(any(feature = "yoloe-visual", feature = "yoloe-pf"))]
use super::config::TaskKind;

impl WasmConfig {
    /// Builds the native detection [`crate::detect::Config`].
    pub(super) fn to_detect_config(&self) -> crate::Result<crate::detect::Config> {
        Ok(crate::detect::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .build())
    }
}

#[cfg(feature = "segment")]
impl WasmConfig {
    /// Builds the native segmentation [`crate::segment::Config`].
    pub(super) fn to_segment_config(&self) -> crate::Result<crate::segment::Config> {
        Ok(crate::segment::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .build())
    }
}

#[cfg(feature = "semantic")]
impl WasmConfig {
    /// Builds the native semantic [`crate::semantic::Config`].
    pub(super) fn to_semantic_config(&self) -> crate::Result<crate::semantic::Config> {
        Ok(crate::semantic::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .build())
    }
}

#[cfg(feature = "classify")]
impl WasmConfig {
    /// Builds the native classify [`crate::classify::Config`].
    pub(super) fn to_classify_config(&self) -> crate::Result<crate::classify::Config> {
        Ok(crate::classify::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .build())
    }
}

#[cfg(feature = "obb")]
impl WasmConfig {
    /// Builds the native OBB [`crate::obb::Config`].
    pub(super) fn to_obb_config(&self) -> crate::Result<crate::obb::Config> {
        Ok(crate::obb::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .build())
    }
}

#[cfg(feature = "pose")]
impl WasmConfig {
    /// Builds the native pose [`crate::pose::Config`].
    pub(super) fn to_pose_config(&self) -> crate::Result<crate::pose::Config> {
        Ok(crate::pose::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size)
            .with_labels_count(self.labels_count)
            .with_keypoints_count(self.keypoints_count)
            .build())
    }
}

#[cfg(any(feature = "yoloe-visual", feature = "yoloe-pf"))]
impl WasmConfig {
    /// Builds the native YOLOE [`crate::yoloe::Config`].
    pub(super) fn to_yoloe_config(&self) -> crate::Result<crate::yoloe::Config> {
        let mut builder = crate::yoloe::config_builder()
            .with_scale(self.scale)
            .with_device(self.device.to_device())
            .with_input_size(self.image_size);
        builder = match self.task {
            TaskKind::YoloePromptFree => builder
                .with_checkpoint(crate::yoloe::usage::CheckpointKind::PromptFree)
                .with_prompt_free(true)
                .with_lrpc(true)
                .with_visual_prompts(false),
            _ => builder,
        };
        Ok(builder.build())
    }
}
