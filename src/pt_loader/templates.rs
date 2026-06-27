//! Embedded `data.pkl` template store.
//!
//! Every official YOLO26 `.pt` is a zip whose `data.pkl` describes the module
//! tree, tensor metadata and storage layout. To write a checkpoint that
//! `torch.load` can read, we reuse the official `data.pkl` verbatim and only
//! regenerate the `data/<N>` storage blobs from the caller's tensors.
//!
//! All 30 `(task, scale)` templates are embedded as a single solid-compressed
//! zstd blob (`templates/templates_blob.zst`, ~600 KiB) so they ship under the
//! `pt` feature only. Each entry records the original archive root (e.g. `best`
//! or `yolo26n-sem`), which differs per file and prefixes every zip entry; the
//! container is a little-endian table of `u32` count then per entry: `u16`
//! name_len + name (`<task>_<scale>`), `u16` root_len + root, `u32` pkl_len +
//! raw `data.pkl` bytes.

use std::io::Read;

use crate::Result;
use crate::model::Scale;

/// The solid-compressed zstd archive shipped at build time.
const TEMPLATES_ZST: &[u8] = include_bytes!("templates/templates_blob.zst");

/// One template entry: the archive root prefixing its entries and the raw pkl.
struct Template {
    root: String,
    pkl: std::ops::Range<usize>,
}

/// Lazily decompressed template table. Decompression runs once per process on
/// first lookup; subsequent lookups hit the cache.
struct TemplateStore {
    blob: Vec<u8>,
    entries: Vec<(String, Template)>,
}

/// Holds the (possibly failed) decompression result so a build-time embed
/// corruption is reported rather than silently producing an empty store.
static STORE: std::sync::OnceLock<std::result::Result<TemplateStore, String>> =
    std::sync::OnceLock::new();

/// A resolved template: its `data.pkl` bytes and the archive directory that
/// prefixes its storage blobs (`<root>/data`).
pub(crate) struct ResolvedTemplate<'a> {
    pub pkl_bytes: &'a [u8],
    pub dir_name: String,
}

/// Returns the `data.pkl` bytes and archive directory for `(task, scale)`.
pub(crate) fn resolve(task: &str, scale: Scale) -> Result<ResolvedTemplate<'static>> {
    let store = STORE.get_or_init(decompress_store);
    let store = store
        .as_ref()
        .map_err(|e| crate::Error::InvalidConfig(e.clone()))?;
    let key = template_key(task, scale);
    let template = store
        .entries
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, t)| t)
        .ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "no embedded .pt template for '{key}' (task={task}, scale={scale})"
            ))
        })?;
    Ok(ResolvedTemplate {
        pkl_bytes: &store.blob[template.pkl.clone()],
        dir_name: format!("{}/data", template.root),
    })
}

/// Builds the canonical template lookup key, e.g. `detect_n`, `classify_x`.
fn template_key(task: &str, scale: Scale) -> String {
    format!("{task}_{scale}")
}

/// Decompresses the embedded zstd blob and indexes the entry table.
fn decompress_store() -> std::result::Result<TemplateStore, String> {
    let mut decoder = ruzstd::decoding::StreamingDecoder::new(TEMPLATES_ZST)
        .map_err(|e| format!("zstd decode failed: {e}"))?;
    let mut blob = Vec::new();
    decoder
        .read_to_end(&mut blob)
        .map_err(|e| format!("zstd decode failed: {e}"))?;
    let entries = parse_table(&blob).map_err(|e| e.to_string())?;
    Ok(TemplateStore { blob, entries })
}

/// Parses the little-endian entry table from the decompressed container.
fn parse_table(blob: &[u8]) -> Result<Vec<(String, Template)>> {
    use std::io::{Cursor, Read};
    let mut cur = Cursor::new(blob);
    let mut buf4 = [0u8; 4];
    cur.read_exact(&mut buf4)?;
    let count = u32::from_le_bytes(buf4) as usize;
    let mut entries = Vec::with_capacity(count);
    let mut buf2 = [0u8; 2];
    for _ in 0..count {
        cur.read_exact(&mut buf2)?;
        let name = read_string(&mut cur, u16::from_le_bytes(buf2) as usize)?;
        cur.read_exact(&mut buf2)?;
        let root = read_string(&mut cur, u16::from_le_bytes(buf2) as usize)?;
        cur.read_exact(&mut buf4)?;
        let pkl_len = u32::from_le_bytes(buf4) as usize;
        let start = cur.position() as usize;
        let pkl = start..start + pkl_len;
        cur.set_position((start + pkl_len) as u64);
        entries.push((name, Template { root, pkl }));
    }
    Ok(entries)
}

/// Reads a length-prefixed UTF-8 string from the cursor.
fn read_string(cur: &mut std::io::Cursor<&[u8]>, len: usize) -> Result<String> {
    use std::io::Read;
    let mut bytes = vec![0u8; len];
    cur.read_exact(&mut bytes)?;
    String::from_utf8(bytes)
        .map_err(|e| crate::Error::InvalidConfig(format!("template string utf8: {e}")))
}
