//! Device utilities.

pub use candle_core::Device;

/// Serializable device descriptor that resolves to a candle [`Device`] by index.
///
/// Unlike [`Device`] itself, which holds live backend handles and therefore
/// cannot cross the wasm/JSON boundary, this is a plain tag describing *which*
/// backend to use. [`DeviceSpec::to_device`] builds the actual candle device,
/// gracefully falling back to the CPU when the requested backend is unavailable
/// in the current build (e.g. CUDA/Metal on a CPU-only or wasm32 target).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSpec {
    /// Host CPU.
    Cpu,
    /// CUDA device at the given ordinal index.
    Cuda(usize),
    /// Metal device at the given index.
    Metal(usize),
}

impl DeviceSpec {
    /// Builds the candle [`Device`] for this spec, falling back to the CPU when
    /// the requested backend is unavailable in the current build.
    ///
    /// This mirrors [`auto`]: CUDA/Metal are only attempted when the matching
    /// cargo feature is enabled, so the same code compiles on CPU-only and
    /// wasm32 targets without the native backends.
    pub fn to_device(self) -> Device {
        match self {
            DeviceSpec::Cpu => Device::Cpu,
            DeviceSpec::Cuda(index) => {
                if cfg!(feature = "cuda") {
                    Device::cuda_if_available(index).unwrap_or(Device::Cpu)
                } else {
                    Device::Cpu
                }
            }
            DeviceSpec::Metal(index) => {
                if cfg!(feature = "metal") {
                    Device::metal_if_available(index).unwrap_or(Device::Cpu)
                } else {
                    Device::Cpu
                }
            }
        }
    }

    /// Builds the candle [`Device`] for this spec, **erroring** when an
    /// explicitly requested CUDA/Metal backend is unavailable in this build.
    ///
    /// Use this for user-explicit device requests so a misconfiguration (e.g.
    /// requesting CUDA on a CPU-only or wasm32 build) surfaces instead of
    /// silently running on CPU. [`Self::to_device`] / [`auto`] remain
    /// best-effort and fall back to CPU.
    pub fn to_device_strict(self) -> crate::Result<Device> {
        match self {
            DeviceSpec::Cpu => Ok(Device::Cpu),
            DeviceSpec::Cuda(index) => {
                if cfg!(feature = "cuda") {
                    Device::cuda_if_available(index).map_err(|err| {
                        crate::Error::InvalidConfig(format!(
                            "CUDA device {index} was requested but is unavailable: {err}"
                        ))
                    })
                } else {
                    Err(crate::Error::InvalidConfig(
                        "CUDA device requested but the `cuda` feature is not enabled".to_string(),
                    ))
                }
            }
            DeviceSpec::Metal(index) => {
                if cfg!(feature = "metal") {
                    Device::metal_if_available(index).map_err(|err| {
                        crate::Error::InvalidConfig(format!(
                            "Metal device {index} was requested but is unavailable: {err}"
                        ))
                    })
                } else {
                    Err(crate::Error::InvalidConfig(
                        "Metal device requested but the `metal` feature is not enabled".to_string(),
                    ))
                }
            }
        }
    }
}

/// Automatically selects the best device available.
///
/// When `cuda` or `metal` features are enabled,
/// the corresponding device at index 0 will be tried first,
/// and the CPU will be used if it is not available.
pub fn auto() -> Device {
    if cfg!(feature = "cuda") {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "metal") {
        Device::metal_if_available(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    }
}
