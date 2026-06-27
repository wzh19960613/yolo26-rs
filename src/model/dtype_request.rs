//! Requested compute dtype for model loading: auto-resolved from the device,
//! target architecture and checkpoint dtype, or fixed by the caller.

use std::path::Path;

use candle_core::{DType, Device};

use crate::Result;

/// User request for the runtime compute dtype of a YOLO26 / YOLOE model.
///
/// The dtype passed to a model loader is **not** the dtype stored in the
/// weights file — it is the precision weights are cast to at load time. With
/// [`DtypeRequest::Auto`] (the default) the loader resolves the compute dtype
/// from **three factors together**: the resolved [`Device`], the target
/// architecture (CPU vs. GPU vs. `wasm32`), and the checkpoint's native dtype.
///
/// # Auto resolution rules (unified across all task roots and YOLOE)
///
/// - **GPU (CUDA / Metal)**: the compute dtype matches the checkpoint dtype.
///   An F16 checkpoint runs in F16 on GPU (faster, lighter, numerically safe);
///   an F32 checkpoint runs in F32.
/// - **CPU**: forced to **F32** even when the checkpoint is F16, because
///   candle's CPU backend emulates F16 matmul (slower than F32 in practice).
///   This is the key fix: the shipped YOLO26 checkpoints are F16, but a CPU
///   deployment should default to F32.
/// - **`wasm32`**: forced to **F32** (SIMD coverage for F16 is limited).
/// - `Fixed(d)` bypasses all of the above and forces `d`.
///
/// `device::auto()` is consulted first by each loader to pick the device, and
/// the resolved dtype then follows the rules above. A caller can override any
/// default with `with_dtype(...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DtypeRequest {
    /// Resolve the compute dtype from the resolved device, target architecture
    /// and the checkpoint's native dtype (see the type-level docs for the
    /// rules). This is the default.
    #[default]
    Auto,
    /// Force weights to be cast to a specific precision at load time, e.g. to
    /// run an F16 checkpoint in F32 or vice versa, ignoring the Auto rules.
    Fixed(DType),
}

impl From<DType> for DtypeRequest {
    fn from(dtype: DType) -> Self {
        Self::Fixed(dtype)
    }
}

impl DtypeRequest {
    /// Convenience constructor for a fixed request.
    pub const fn fixed(dtype: DType) -> Self {
        Self::Fixed(dtype)
    }

    /// Resolves the compute dtype from the resolved device, target
    /// architecture and a checkpoint-native dtype.
    ///
    /// This is the unified Auto rule shared by every task root and YOLOE: GPU
    /// follows the checkpoint dtype, CPU / `wasm32` force F32. Callers that
    /// have already resolved the device (e.g. from `device::auto()` or a
    /// `Config::device`) pass it here so the dtype can track it.
    pub fn resolve(&self, device: &Device, weight_dtype: DType) -> DType {
        match self {
            Self::Fixed(dtype) => *dtype,
            Self::Auto => Self::auto_resolve(device, weight_dtype),
        }
    }

    /// Auto resolution for a checkpoint-native dtype on a resolved device.
    ///
    /// GPU (CUDA / Metal) keeps the checkpoint dtype; CPU and `wasm32` force
    /// F32 so F16 checkpoints do not fall into candle's slow CPU F16 emulation.
    fn auto_resolve(device: &Device, weight_dtype: DType) -> DType {
        if Self::device_is_cpu_or_wasm(device) {
            DType::F32
        } else {
            weight_dtype
        }
    }

    /// Returns `true` when the device is CPU or the build targets `wasm32`
    /// (which is always CPU under the hood and lacks F16 SIMD support).
    fn device_is_cpu_or_wasm(device: &Device) -> bool {
        if cfg!(target_arch = "wasm32") {
            return true;
        }
        matches!(device, Device::Cpu)
    }

    /// Resolves the compute dtype against a SafeTensors byte buffer and a
    /// resolved [`Device`].
    ///
    /// `Auto` reads the dtype of the first tensor in the SafeTensors header
    /// and then applies the device/arch rules above. `Fixed(d)` returns `d`
    /// unchanged. An empty buffer is an error.
    pub fn resolve_safetensors(&self, weights: &[u8], device: &Device) -> Result<DType> {
        use candle_core::safetensors::Load;
        match self {
            Self::Fixed(dtype) => Ok(*dtype),
            Self::Auto => {
                let safetensors = candle_core::safetensors::SliceSafetensors::new(weights)?;
                let weight_dtype = safetensors
                    .tensors()
                    .into_iter()
                    .next()
                    .map(|(_, view)| view.load(&candle_core::Device::Cpu).map(|t| t.dtype()))
                    .ok_or_else(|| {
                        crate::Error::InvalidConfig(
                            "cannot infer dtype: SafeTensors checkpoint has no tensors".to_string(),
                        )
                    })??;
                Ok(self.resolve(device, weight_dtype))
            }
        }
    }

    /// Resolves the compute dtype against an official `.pt` checkpoint and a
    /// resolved [`Device`].
    ///
    /// `Auto` reads the dtype of the first tensor in the archive and then
    /// applies the device/arch rules above. `Fixed(d)` returns `d` unchanged.
    #[cfg(feature = "pt")]
    pub fn resolve_pt(&self, path: impl AsRef<Path>, device: &Device) -> Result<DType> {
        match self {
            Self::Fixed(dtype) => Ok(*dtype),
            Self::Auto => {
                let load_device = candle_core::Device::Cpu;
                let tensors = crate::pt_loader::load_pt_to_tensors(path, &load_device)?;
                let weight_dtype = tensors
                    .into_iter()
                    .next()
                    .map(|(_, tensor)| tensor.dtype())
                    .ok_or_else(|| {
                        crate::Error::InvalidConfig(
                            "cannot infer dtype: .pt checkpoint has no tensors".to_string(),
                        )
                    })?;
                Ok(self.resolve(device, weight_dtype))
            }
        }
    }

    /// Resolves the compute dtype against an in-memory `.pt` byte buffer and a
    /// resolved [`Device`], mirroring [`Self::resolve_pt`].
    ///
    /// `Auto` reads the dtype of the first tensor in the archive and then
    /// applies the device/arch rules above. `Fixed(d)` returns `d` unchanged.
    #[cfg(feature = "pt")]
    pub fn resolve_pt_bytes(&self, bytes: &[u8], device: &Device) -> Result<DType> {
        match self {
            Self::Fixed(dtype) => Ok(*dtype),
            Self::Auto => {
                let load_device = candle_core::Device::Cpu;
                let tensors = crate::pt_loader::load_pt_to_tensors_from_bytes(bytes, &load_device)?;
                let weight_dtype = tensors
                    .into_iter()
                    .next()
                    .map(|(_, tensor)| tensor.dtype())
                    .ok_or_else(|| {
                        crate::Error::InvalidConfig(
                            "cannot infer dtype: .pt checkpoint has no tensors".to_string(),
                        )
                    })?;
                Ok(self.resolve(device, weight_dtype))
            }
        }
    }

    /// Returns a concrete dtype without a checkpoint, defaulting `Auto` to
    /// `F32`. Useful for tests that build a config without loading weights.
    pub fn resolve_or_f32(&self) -> DType {
        match self {
            Self::Fixed(dtype) => *dtype,
            Self::Auto => DType::F32,
        }
    }
}
