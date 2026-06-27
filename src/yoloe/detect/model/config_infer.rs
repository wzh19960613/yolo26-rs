//! Identity shape inference for [`Config`].
//!
//! Extracted from [`crate::yoloe::detect::model::config`]: the `infer_from_*` loaders
//! read the official SafeTensors/`.pt` tensor shapes (delegating to
//! [`crate::yoloe::detect::model::shape`]) to build a config. Low-level
//! callers can pass raw scale/device/dtype args; user-facing loaders infer the
//! scale from checkpoint shapes and take device/dtype from
//! [`Config`](crate::yoloe::config::Config).

use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

use candle_core::Device;

use crate::Scale;
use crate::model::DtypeRequest;

use crate::yoloe::checkpoint::layout::Layout;
use crate::yoloe::detect::model::config::Config;
use crate::yoloe::detect::model::shape::{InferredDetectShapes, infer_detect_shapes};
use crate::yoloe::head::contrastive::Contrastive;

impl Config {
    /// Infers the embedding dimension and LRPC availability from SafeTensors bytes.
    pub fn infer_from_safetensors_bytes(
        bytes: &[u8],
        scale: Scale,
        device: Device,
        dtype: impl Into<DtypeRequest>,
        max_predictions: usize,
    ) -> crate::Result<Self> {
        let shapes = shapes_from_safetensors_bytes(bytes)?;
        Self::infer_from_shapes(shapes, scale, device, dtype.into(), max_predictions)
    }

    /// Infers the embedding dimension and LRPC availability from a SafeTensors file.
    pub fn infer_from_safetensors_file(
        path: impl AsRef<Path>,
        scale: Scale,
        device: Device,
        dtype: impl Into<DtypeRequest>,
        max_predictions: usize,
    ) -> crate::Result<Self> {
        Self::infer_from_safetensors_bytes(
            &std::fs::read(path)?,
            scale,
            device,
            dtype,
            max_predictions,
        )
    }

    /// Infers the embedding dimension and LRPC availability from an official
    /// `.pt` checkpoint, mirroring [`Self::infer_from_safetensors_file`].
    #[cfg(feature = "pt")]
    pub fn infer_from_pt_file(
        path: impl AsRef<Path>,
        scale: Scale,
        device: Device,
        dtype: impl Into<DtypeRequest>,
        max_predictions: usize,
    ) -> crate::Result<Self> {
        let shapes = shapes_from_pt(path, &device)?;
        Self::infer_from_shapes(shapes, scale, device, dtype.into(), max_predictions)
    }

    /// Infers head dimensions from SafeTensors bytes, taking device/dtype/
    /// max_predictions from a user-facing
    /// [`Config`](crate::yoloe::config::Config).
    pub(crate) fn infer_from_safetensors_bytes_with_config(
        bytes: &[u8],
        config: &crate::yoloe::config::Config,
    ) -> crate::Result<Self> {
        let shapes = shapes_from_safetensors_bytes(bytes)?;
        Self::infer_from_shapes_with_config(&shapes, config)
    }

    /// Infers head dimensions from an official `.pt` checkpoint, taking
    /// device/dtype/max_predictions from a user-facing
    /// [`Config`](crate::yoloe::config::Config).
    #[cfg(feature = "pt")]
    pub(crate) fn infer_from_pt_file_with_config(
        path: impl AsRef<Path>,
        config: &crate::yoloe::config::Config,
    ) -> crate::Result<Self> {
        let shapes = shapes_from_pt(path, &config.device)?;
        Self::infer_from_shapes_with_config(&shapes, config)
    }

    /// Infers head dimensions from an in-memory `.pt` byte buffer, taking
    /// device/dtype/max_predictions from a user-facing
    /// [`Config`](crate::yoloe::config::Config).
    #[cfg(feature = "pt")]
    pub(crate) fn infer_from_pt_bytes_with_config(
        bytes: &[u8],
        config: &crate::yoloe::config::Config,
    ) -> crate::Result<Self> {
        let shapes = shapes_from_pt_bytes(bytes)?;
        Self::infer_from_shapes_with_config(&shapes, config)
    }

    fn infer_from_shapes(
        shapes: BTreeMap<String, Vec<usize>>,
        scale: Scale,
        device: Device,
        dtype: DtypeRequest,
        max_predictions: usize,
    ) -> crate::Result<Self> {
        let layout = Layout::from_tensor_names(shapes.keys().cloned().collect::<Vec<_>>());
        if !layout.can_load_detect() {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE detection checkpoint is missing key families: {}",
                layout.missing_detect().join(", ")
            )));
        }
        Self::build_from_inferred(
            infer_detect_shapes(&shapes, &layout)?,
            scale,
            device,
            dtype,
            max_predictions,
        )
    }

    pub(crate) fn infer_from_shapes_with_config(
        shapes: &BTreeMap<String, Vec<usize>>,
        config: &crate::yoloe::config::Config,
    ) -> crate::Result<Self> {
        let layout = Layout::from_tensor_names(shapes.keys().cloned().collect::<Vec<_>>());
        if !layout.can_load_detect() {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE detection checkpoint is missing key families: {}",
                layout.missing_detect().join(", ")
            )));
        }
        let inferred = infer_detect_shapes(shapes, &layout)?;
        let scale = config_scale(shapes)?;
        let model = Self {
            scale,
            device: config.device.clone(),
            dtype: config.dtype,
            max_predictions: config.max_predictions,
            embed_dim: inferred.embed_dim,
            official_lrpc: inferred.official_lrpc,
            official_savpe: inferred.official_savpe,
            savpe_hidden: inferred.savpe_hidden,
            prompt_head: inferred.prompt_head,
            cls_hidden: inferred.cls_hidden,
            box_hidden: inferred.box_hidden,
            contrastive: Contrastive::default(),
        };
        model.validate()?;
        Ok(model)
    }

    fn build_from_inferred(
        inferred: InferredDetectShapes,
        scale: Scale,
        device: Device,
        dtype: DtypeRequest,
        max_predictions: usize,
    ) -> crate::Result<Self> {
        let model = Self {
            scale,
            device,
            dtype,
            max_predictions,
            embed_dim: inferred.embed_dim,
            official_lrpc: inferred.official_lrpc,
            official_savpe: inferred.official_savpe,
            savpe_hidden: inferred.savpe_hidden,
            prompt_head: inferred.prompt_head,
            cls_hidden: inferred.cls_hidden,
            box_hidden: inferred.box_hidden,
            contrastive: Contrastive::default(),
        };
        model.validate()?;
        Ok(model)
    }
}

fn config_scale(shapes: &BTreeMap<String, Vec<usize>>) -> crate::Result<Scale> {
    let shapes = shapes
        .iter()
        .map(|(name, dims)| (name.clone(), dims.clone()))
        .collect::<HashMap<_, _>>();
    crate::model::infer_scale_from_shapes(&shapes)
}

fn shapes_from_safetensors_bytes(bytes: &[u8]) -> crate::Result<BTreeMap<String, Vec<usize>>> {
    let safetensors = candle_core::safetensors::SliceSafetensors::new(bytes)?;
    Ok(safetensors
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, view.shape().to_vec()))
        .collect())
}

#[cfg(feature = "pt")]
fn shapes_from_pt(
    path: impl AsRef<Path>,
    device: &Device,
) -> crate::Result<BTreeMap<String, Vec<usize>>> {
    let tensors = crate::pt_loader::load_pt_to_tensors(path, device)?;
    Ok(tensors
        .into_iter()
        .map(|(name, tensor)| (name, tensor.shape().dims().to_vec()))
        .collect())
}

#[cfg(feature = "pt")]
fn shapes_from_pt_bytes(bytes: &[u8]) -> crate::Result<BTreeMap<String, Vec<usize>>> {
    let tensors = crate::pt_loader::load_pt_to_tensors_from_bytes(bytes, &Device::Cpu)?;
    Ok(tensors
        .into_iter()
        .map(|(name, tensor)| (name, tensor.shape().dims().to_vec()))
        .collect())
}
