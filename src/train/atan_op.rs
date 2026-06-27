//! A differentiable `atan` tensor op for candle 0.10.2.
//!
//! Candle 0.10.2 ships `exp/log/sin/cos/tanh/sqr/sqrt` unary ops but **no
//! `atan`**. The official YOLO26 CIoU box loss needs
//! `atan(width / height)` for the aspect-ratio consistency term, and its
//! gradient must flow through backprop. We register a `CustomOp1` whose forward
//! evaluates `atan` exactly (matching PyTorch) and whose backward uses the
//! closed form `1 / (1 + x^2)`, expressed via existing candle ops so it runs on
//! every backend.

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::cpu_backend::unary_map;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, Error, Layout, MetalStorage, Result, Shape, Tensor,
};

/// Elementwise `atan(x)` with exact forward and exact `1/(1+x^2)` backward.
pub(crate) fn atan(tensor: &Tensor) -> Result<Tensor> {
    tensor.apply_op1(Atan)
}

struct Atan;

fn atan_fwd_cpu(s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
    // YOLO26 box regression always runs in F32; support F32 and F64 exactly
    // and reject other dtypes (CIoU never receives F16/BF16 from the loss path).
    let out = match s {
        CpuStorage::F32(data) => CpuStorage::F32(unary_map(data, l, f32::atan)),
        CpuStorage::F64(data) => CpuStorage::F64(unary_map(data, l, f64::atan)),
        other => {
            return Err(Error::UnsupportedDTypeForOp(other.dtype(), "atan").bt());
        }
    };
    Ok((out, l.shape().clone()))
}

impl CustomOp1 for Atan {
    fn name(&self) -> &'static str {
        "atan"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        atan_fwd_cpu(s, l)
    }

    fn cuda_fwd(&self, _storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        Err(candle_core::Error::Cuda(
            "no cuda implementation for atan (use CPU or Metal)".into(),
        ))
    }

    fn metal_fwd(&self, storage: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        // Round-trip Metal -> CPU -> atan -> CPU -> Metal. Slower than a native
        // kernel but exact and backend-portable; atan only runs on the small
        // aspect-ratio tensors inside CIoU, not on the main feature maps.
        let cpu = storage.to_cpu_storage()?;
        let (out_cpu, shape) = atan_fwd_cpu(&cpu, layout)?;
        let out_metal = storage.device().storage_from_cpu_storage(&out_cpu)?;
        Ok((out_metal, shape))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx atan(x) = 1 / (1 + x^2).
        let ones = arg.ones_like()?;
        let denom = ones.broadcast_add(&arg.sqr()?)?;
        let grad = grad_res.broadcast_div(&denom)?;
        Ok(Some(grad))
    }
}
