//! Generates `data/<N>` storage blob bytes from a tensor map, following the
//! storage layout recorded in a parsed `data.pkl` template.
//!
//! Each tensor info names its blob (`<dir>/data/<N>`), a byte offset within
//! that blob, a dtype and a shape. We materialize every blob the template
//! references — including storages outside the `model` subtree (EMA / optimizer
//! / misc buffers) — so `torch.load` finds every `data/<N>` entry it expects.
//! Model tensors present in the caller's map are serialized (transposing
//! Fortran-contiguous layouts back to PyTorch order); every other storage is
//! zero-filled so the layout stays intact.

use std::collections::HashMap;

use candle_core::pickle::TensorInfo;
use candle_core::{DType, Device, Tensor};

use crate::Result;

use super::reader::ParsedCheckpoint;

/// `(blob_path, offset_within_blob)` — the write address of one tensor.
type BlobAddr = (String, usize);

/// Resolved bytes for one tensor: its blob address and the little-endian bytes
/// to write there. Missing tensors contribute a zero-filled region.
struct ResolvedTensor {
    addr: BlobAddr,
    bytes: Vec<u8>,
}

/// Builds the full set of storage blobs as `(blob_path, blob_bytes)` pairs,
/// ready to be written into the destination zip.
///
/// `named_infos` carries the `model`-subtree tensors that map to `tensors` by
/// dotted name; `all_infos` lists every storage the pickle references (including
/// non-model storages) so we emit a blob per referenced `data/<N>`. Non-model
/// storages and any model tensor absent from `tensors` (e.g.
/// `num_batches_tracked`) are zero-filled.
pub(crate) fn build_blobs(
    named_infos: &[(String, TensorInfo)],
    all_infos: &[TensorInfo],
    tensors: &HashMap<String, Tensor>,
) -> Result<Vec<(String, Vec<u8>)>> {
    build_blobs_inner(named_infos, all_infos, tensors, None)
}

pub(crate) fn build_blobs_with_fallback(
    named_infos: &[(String, TensorInfo)],
    all_infos: &[TensorInfo],
    tensors: &HashMap<String, Tensor>,
    fallback: &ParsedCheckpoint,
) -> Result<Vec<(String, Vec<u8>)>> {
    build_blobs_inner(named_infos, all_infos, tensors, Some(fallback))
}

fn build_blobs_inner(
    named_infos: &[(String, TensorInfo)],
    all_infos: &[TensorInfo],
    tensors: &HashMap<String, Tensor>,
    fallback: Option<&ParsedCheckpoint>,
) -> Result<Vec<(String, Vec<u8>)>> {
    // path -> tensor data bytes, for the model storages we can resolve.
    let mut path_bytes: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, info) in named_infos {
        let bytes = match tensors.get(name) {
            Some(tensor) => serialize_tensor(tensor, info, name)?,
            None => continue,
        };
        path_bytes.insert(info.path.clone(), bytes);
    }
    let resolved = resolve_all(all_infos, &path_bytes, fallback)?;
    let extents = compute_extents(&resolved);
    let mut blobs: HashMap<String, Vec<u8>> = extents
        .iter()
        .map(|(path, len)| (path.clone(), vec![0u8; *len]))
        .collect();
    for rt in resolved {
        let blob = blobs
            .get_mut(&rt.addr.0)
            .expect("extent pre-allocated every blob");
        let end = rt.addr.1 + rt.bytes.len();
        blob[rt.addr.1..end].copy_from_slice(&rt.bytes);
    }
    Ok(blobs.into_iter().collect())
}

/// Resolves every storage info into a `(addr, bytes)` pair, using the precomputed
/// `path_bytes` for model storages and zero-filling everything else.
fn resolve_all(
    all_infos: &[TensorInfo],
    path_bytes: &HashMap<String, Vec<u8>>,
    fallback: Option<&ParsedCheckpoint>,
) -> Result<Vec<ResolvedTensor>> {
    let mut out = Vec::with_capacity(all_infos.len());
    for info in all_infos {
        let numel = info.layout.shape().elem_count();
        let n_bytes = numel
            .checked_mul(info.dtype.size_in_bytes())
            .ok_or_else(|| crate::Error::InvalidTensor("storage too large".to_string()))?;
        let bytes = match path_bytes.get(&info.path) {
            Some(bytes) => bytes.clone(),
            None => match fallback {
                Some(parsed) => super::tensor_build::read_storage_bytes(
                    parsed,
                    &info.path,
                    info.layout.start_offset(),
                    n_bytes,
                )?,
                None => vec![0u8; n_bytes],
            },
        };
        if bytes.len() < n_bytes {
            return Err(crate::Error::InvalidTensor(format!(
                "storage '{}' byte count {} < expected {n_bytes}",
                info.path,
                bytes.len()
            )));
        }
        out.push(ResolvedTensor {
            addr: (info.path.clone(), info.layout.start_offset()),
            bytes,
        });
    }
    Ok(out)
}

/// Computes the byte length each blob must have to fit all of its tensors.
fn compute_extents(resolved: &[ResolvedTensor]) -> Vec<(String, usize)> {
    let mut extents: HashMap<String, usize> = HashMap::new();
    for rt in resolved {
        let end = rt.addr.1 + rt.bytes.len();
        extents
            .entry(rt.addr.0.clone())
            .and_modify(|e| *e = (*e).max(end))
            .or_insert(end);
    }
    extents.into_iter().collect()
}

/// Serializes one tensor into the little-endian byte order PyTorch expects,
/// transposing Fortran-contiguous weights back to column-major storage order.
///
/// The dtype is dictated by the template's storage layout: if the caller's
/// tensor uses a different precision (e.g. an F32 training variable written
/// against an F16 template), it is cast to the template dtype before encoding.
fn serialize_tensor(tensor: &Tensor, info: &TensorInfo, name: &str) -> Result<Vec<u8>> {
    let info_dims = info.layout.dims();
    let tensor_dims: Vec<usize> = tensor.dims().to_vec();
    let on_cpu = tensor.to_device(&Device::Cpu)?;
    let cast = if on_cpu.dtype() == info.dtype {
        on_cpu
    } else {
        on_cpu.to_dtype(info.dtype)?
    };
    // Official checkpoints occasionally store the same tensor at different
    // ranks (e.g. `[4585, 80]` vs `[4585, 80, 1, 1]`). When the element counts
    // match, reshape to the template rank rather than failing.
    let aligned = if tensor_dims != info_dims {
        let tensor_elems: usize = tensor_dims.iter().product();
        let info_elems: usize = info_dims.iter().product();
        if tensor_elems == info_elems {
            cast.reshape(info_dims)?
        } else {
            return Err(crate::Error::InvalidTensor(format!(
                "tensor '{name}' shape mismatch: template has {:?}, tensor has {:?}",
                info_dims, tensor_dims
            )));
        }
    } else {
        cast
    };
    let flat = if info.layout.is_fortran_contiguous() && info_dims.len() > 1 {
        let rank = info_dims.len();
        let perm: Vec<usize> = (0..rank).rev().collect();
        aligned.permute(perm)?.flatten_all()?
    } else {
        aligned.flatten_all()?
    };
    tensor_to_bytes(&flat, info.dtype)
}

/// Extracts the raw little-endian bytes of a flattened 1-D tensor by dtype.
fn tensor_to_bytes(tensor: &Tensor, dtype: DType) -> Result<Vec<u8>> {
    match dtype {
        DType::F16 => {
            let v = tensor.to_vec1::<half::f16>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::BF16 => {
            let v = tensor.to_vec1::<half::bf16>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::F32 => {
            let v = tensor.to_vec1::<f32>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::F64 => {
            let v = tensor.to_vec1::<f64>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::U8 => Ok(tensor.to_vec1::<u8>()?),
        DType::U32 => {
            let v = tensor.to_vec1::<u32>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        DType::I64 => {
            let v = tensor.to_vec1::<i64>()?;
            Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        other => Err(crate::Error::InvalidTensor(format!(
            "unsupported dtype for pt write: {other:?}"
        ))),
    }
}
