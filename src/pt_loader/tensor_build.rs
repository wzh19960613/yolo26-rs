//! Materializes flattened tensor infos into `candle_core::Tensor`s by reading
//! the raw storage blobs from the PyTorch zip container.
//!
//! Each tensor info points at `<dir_name>/data/<key>`. The blob holds the
//! storage bytes (possibly with an offset). We handle the common contiguous
//! layout directly and transpose Fortran-contiguous tensors to match PyTorch
//! semantics, mirroring candle's own `PthTensors::get`.

use std::collections::HashMap;
use std::io::Read;

use candle_core::{Device, Tensor};

use crate::Result;

use super::reader::ParsedCheckpoint;

/// Builds the tensor map by re-opening the checkpoint zip and reading each
/// storage blob on demand. Re-opening per call (rather than holding the file
/// open) mirrors candle's `PthTensors` and keeps the API ownership simple.
pub fn build_tensor_map(
    parsed: &ParsedCheckpoint,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(parsed.tensor_infos.len());
    for (name, info) in &parsed.tensor_infos {
        let numel: usize = info.layout.shape().elem_count();
        let n_bytes = numel
            .checked_mul(info.dtype.size_in_bytes())
            .ok_or_else(|| crate::Error::InvalidTensor(format!("tensor {name} too large")))?;
        let bytes = read_storage_bytes(parsed, &info.path, info.layout.start_offset(), n_bytes)?;
        let dims: Vec<usize> = info.layout.dims().to_vec();
        let rank = dims.len();
        let tensor = Tensor::from_raw_buffer(&bytes, info.dtype, &dims, device)?;
        let tensor = if rank > 1 && info.layout.is_fortran_contiguous() {
            let reversed: Vec<usize> = dims.iter().rev().copied().collect();
            let tensor = tensor.reshape(reversed)?;
            let perm: Vec<usize> = (0..rank).rev().collect();
            tensor.permute(perm)?
        } else {
            tensor
        };
        out.insert(name.clone(), tensor);
    }
    Ok(out)
}

/// Reads exactly `n_bytes` bytes starting at `start_offset` (in bytes) from a
/// storage blob in the archive. Reading only the tensor's own bytes keeps
/// shared-storage views correct even when the blob is larger.
pub(crate) fn read_storage_bytes(
    parsed: &ParsedCheckpoint,
    blob_path: &str,
    start_offset: usize,
    n_bytes: usize,
) -> Result<Vec<u8>> {
    // Prefer the in-memory byte buffer when present (wasm path), otherwise fall
    // back to re-opening the source file (native path). An in-memory template
    // parsed by `read_checkpoint_from_bytes` carries neither and must never reach
    // this point.
    if let Some(bytes) = parsed.source_bytes.as_ref() {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes.as_ref()))?;
        read_blob(&mut zip, blob_path, start_offset, n_bytes)
    } else {
        let source_path = parsed.source_path.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "cannot read storage bytes from an in-memory checkpoint template".to_string(),
            )
        })?;
        let file = std::fs::File::open(source_path)?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
        read_blob(&mut zip, blob_path, start_offset, n_bytes)
    }
}

/// Reads `n_bytes` from a single storage blob inside an already-open zip,
/// skipping `start_offset` bytes first. Shared by the file-backed and
/// bytes-backed paths so the seek/read logic is written once.
fn read_blob<R: std::io::Read + std::io::Seek>(
    zip: &mut zip::ZipArchive<R>,
    blob_path: &str,
    start_offset: usize,
    n_bytes: usize,
) -> Result<Vec<u8>> {
    let mut reader = std::io::BufReader::new(zip.by_name(blob_path)?);
    if start_offset > 0 {
        let mut sink = std::io::sink();
        std::io::copy(&mut reader.by_ref().take(start_offset as u64), &mut sink)?;
    }
    let mut buf = vec![0u8; n_bytes];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}
