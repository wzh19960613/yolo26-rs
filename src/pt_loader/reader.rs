//! Reads the PyTorch zip container and parses its pickle object graph.
//!
//! Reuses [`candle_core::pickle::Stack`] for the actual pickle opcode stream
//! (protocol 2 with the persistent-load storage references that PyTorch uses),
//! then locates the `model` (or EMA) module object together with the directory
//! name that prefixes every storage blob (`<archive_root>/data/<key>`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::pickle::{Object, Stack, TensorInfo};

use crate::Result;

/// Parsed checkpoint: the chosen module object plus the archive directory that
/// prefixes its storage blobs and the set of flattened tensor infos.
pub struct ParsedCheckpoint {
    /// Source file path, kept so storage blobs can be read on demand.
    /// `None` when parsed from an in-memory template that has no source file.
    pub(crate) source_path: Option<PathBuf>,
    /// In-memory copy of the full `.pt` zip bytes, used when the checkpoint was
    /// parsed from a byte buffer (e.g. the wasm API) instead of a file path.
    /// Mutually exclusive with [`source_path`](Self::source_path): exactly one
    /// is `Some` for a checkpoint whose storage blobs must be readable.
    pub(crate) source_bytes: Option<Arc<[u8]>>,
    /// Archive directory prefix, e.g. `best/data`.
    #[allow(dead_code)]
    pub dir_name: String,
    /// Flattened dotted-name tensor infos extracted from the module tree.
    pub tensor_infos: Vec<(String, TensorInfo)>,
    /// Top-level metadata object (a shallow copy of the `data.pkl` dict).
    pub top_level: Object,
    /// Full deserialized top-level object graph, kept for whole-graph scans
    /// (e.g. enumerating every storage the pickle references for `.pt` writing).
    pub(crate) full_object: Object,
    /// Shallow copy of the wrapped module's `__setstate__` dict, used to read
    /// task-level metadata such as class names that live on the module.
    pub module_state: Object,
}

/// Parses a `.pt` file, preferring the `model` module and falling back to `ema`
/// when the training checkpoint stores the averaged weights there instead.
pub fn read_checkpoint(path: &Path) -> Result<ParsedCheckpoint> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let pkl_name = zip
        .file_names()
        .map(|f| f.to_string())
        .find(|f| f.ends_with("data.pkl"))
        .ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "{} is not a PyTorch .pt checkpoint (no data.pkl entry)",
                path.display()
            ))
        })?;

    // Storage blobs live next to data.pkl under `data/`, so the per-tensor
    // path is `<archive_root>/data/<key>`. We strip the `.pkl` suffix to get
    // `<archive_root>/data`, matching candle's PthTensors path construction.
    let dir_name = pkl_name
        .strip_suffix(".pkl")
        .map(|s| s.to_string())
        .unwrap_or_default();

    let mut reader = std::io::BufReader::new(zip.by_name(&pkl_name)?);
    let object = parse_pickle_bytes(&mut reader)?;
    let parsed = build_parsed(object, dir_name)?;
    Ok(ParsedCheckpoint {
        source_path: Some(path.to_path_buf()),
        source_bytes: None,
        dir_name: parsed.dir_name,
        tensor_infos: parsed.tensor_infos,
        top_level: parsed.top_level,
        full_object: parsed.full_object,
        module_state: parsed.module_state,
    })
}

/// Parses an in-memory `data.pkl` template. `dir_name` is the archive root that
/// prefixes storage blobs (`<dir_name>/data/<key>`); for YOLO26 checkpoints this
/// is the path of the `data.pkl` entry without the `.pkl` suffix.
///
/// The returned checkpoint carries no `source_path`: storage bytes are not read
/// from disk for a template, so callers must never reach `source_path`.
pub(crate) fn read_checkpoint_from_bytes(
    pkl_bytes: &[u8],
    dir_name: &str,
) -> Result<ParsedCheckpoint> {
    let mut reader = std::io::BufReader::new(pkl_bytes);
    let object = parse_pickle_bytes(&mut reader)?;
    let parsed = build_parsed(object, dir_name.to_string())?;
    Ok(ParsedCheckpoint {
        source_path: None,
        source_bytes: None,
        dir_name: parsed.dir_name,
        tensor_infos: parsed.tensor_infos,
        top_level: parsed.top_level,
        full_object: parsed.full_object,
        module_state: parsed.module_state,
    })
}

/// Parses a full `.pt` checkpoint from an in-memory byte buffer, mirroring
/// [`read_checkpoint`] but reading the zip container straight out of memory
/// instead of a file path. Used by the wasm API (no filesystem) and any other
/// caller that already holds the bytes.
///
/// The byte buffer is retained (`Arc<[u8]>`) so per-tensor storage blobs can be
/// read on demand by [`super::tensor_build::build_tensor_map`].
pub fn read_checkpoint_from_zip_bytes(bytes: &[u8]) -> Result<ParsedCheckpoint> {
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes))?;
    let pkl_name = zip
        .file_names()
        .map(|f| f.to_string())
        .find(|f| f.ends_with("data.pkl"))
        .ok_or_else(|| {
            crate::Error::InvalidConfig(
                "byte buffer is not a PyTorch .pt checkpoint (no data.pkl entry)".to_string(),
            )
        })?;
    let dir_name = pkl_name
        .strip_suffix(".pkl")
        .map(|s| s.to_string())
        .unwrap_or_default();

    let mut reader = std::io::BufReader::new(zip.by_name(&pkl_name)?);
    let object = parse_pickle_bytes(&mut reader)?;
    let parsed = build_parsed(object, dir_name)?;
    Ok(ParsedCheckpoint {
        source_path: None,
        source_bytes: Some(Arc::from(bytes)),
        dir_name: parsed.dir_name,
        tensor_infos: parsed.tensor_infos,
        top_level: parsed.top_level,
        full_object: parsed.full_object,
        module_state: parsed.module_state,
    })
}

/// Runs the pickle stack over a buffered reader and returns the top-level object.
fn parse_pickle_bytes<R: std::io::BufRead>(reader: &mut R) -> Result<Object> {
    let mut stack = Stack::empty();
    stack.read_loop(reader)?;
    Ok(stack.finalize()?)
}

/// Shared parsing of the top-level object into a `ParsedCheckpoint` body. The
/// caller fills in `source_path` afterwards.
fn build_parsed(object: Object, dir_name: String) -> Result<ParsedCheckpointBody> {
    let top_level = clone_shallow_top(&object)?;
    let module = pick_module(&object)?;
    let module_state = module_state_shallow(&module);
    // Official YOLO26 checkpoints wrap the actual backbone inside a single
    // `_modules["model"]` container whose keys are the layer indices (0..23).
    // The safetensors layout keys the tensors as `model.0.conv.weight`, so we
    // unwrap that wrapper and flatten the backbone with the `model` prefix.
    let backbone = unwrap_model_wrapper(&module);
    let tensor_infos = super::flatten::flatten_module(&backbone, &dir_name, "model")?;
    Ok(ParsedCheckpointBody {
        dir_name,
        tensor_infos,
        top_level,
        full_object: object,
        module_state,
    })
}

/// Internal body returned by [`build_parsed`], split out so `read_checkpoint` and
/// [`read_checkpoint_from_bytes`] can attach a `source_path` independently.
struct ParsedCheckpointBody {
    dir_name: String,
    tensor_infos: Vec<(String, TensorInfo)>,
    top_level: Object,
    full_object: Object,
    module_state: Object,
}

/// Extracts a shallow copy of the top-level dict for metadata inspection. The
/// full object graph is consumed by module flattening, so we clone only the
/// scalar metadata fields we care about into a fresh `Dict`.
fn clone_shallow_top(object: &Object) -> Result<Object> {
    let entries = match object {
        Object::Dict(entries) => entries,
        _ => return Ok(Object::None),
    };
    let cloned = entries
        .iter()
        .filter(|(key, _)| matches!(key, Object::Unicode(_)))
        .filter_map(|(key, value)| {
            let lightweight = shallow_value(value)?;
            Some((key.clone(), lightweight))
        })
        .collect();
    Ok(Object::Dict(cloned))
}

/// Keeps only cheap metadata values needed by [`super::metadata::extract`]:
/// scalars plus the small `names` dict/list. Tensors and module subgraphs are
/// replaced with `None` so we never clone large object graphs.
fn shallow_value(value: &Object) -> Option<Object> {
    match value {
        Object::Unicode(_)
        | Object::Int(_)
        | Object::Long(_)
        | Object::Float(_)
        | Object::Bool(_)
        | Object::None => Some(value.clone()),
        Object::Dict(_) | Object::List(_) => Some(value.clone()),
        _ => Some(Object::None),
    }
}

/// Selects the module object to flatten: prefers `model`, then `ema`.
fn pick_module(object: &Object) -> Result<Object> {
    let entries = match object {
        Object::Dict(entries) => entries,
        _ => return Ok(object.clone()),
    };
    for key_name in ["model", "ema"] {
        if let Some((_, value)) = entries
            .iter()
            .find(|(key, _)| matches!(key, Object::Unicode(s) if s == key_name))
        {
            return Ok(value.clone());
        }
    }
    Ok(object.clone())
}

/// Descends into a module's single `_modules["model"]` child when present.
/// Official YOLO26 checkpoints store the backbone (layers 0..N) under this
/// wrapper, so unwrapping it produces flat keys like `model.0.conv.weight`.
fn unwrap_model_wrapper(module: &Object) -> Object {
    let state = match module_state_entries(module) {
        Some(state) => state,
        None => return module.clone(),
    };
    let modules_entry = state
        .iter()
        .find(|(k, _)| matches!(k, Object::Unicode(s) if s == "_modules"));
    let inner = match modules_entry {
        Some((_, Object::Dict(pairs))) => pairs,
        _ => return module.clone(),
    };
    let wrapped = inner
        .iter()
        .find(|(k, _)| matches!(k, Object::Unicode(s) if s == "model"));
    match wrapped {
        Some((_, child)) => child.clone(),
        None => module.clone(),
    }
}

/// Returns the module's `__setstate__` dict entries if `object` is a module
/// (a `BUILD`/`REDUCE` over a `Class` whose args is a dict).
fn module_state_entries(object: &Object) -> Option<&Vec<(Object, Object)>> {
    let args = match object {
        Object::Build { callable, args } => {
            let inner = match &**callable {
                Object::Reduce { callable, .. } => &**callable,
                _ => return None,
            };
            match inner {
                Object::Class { .. } => &**args,
                _ => return None,
            }
        }
        Object::Reduce { callable, args } => match &**callable {
            Object::Class { .. } => &**args,
            _ => return None,
        },
        _ => return None,
    };
    match args {
        Object::Dict(state) => Some(state),
        _ => None,
    }
}

/// Builds a shallow copy of the module's state dict, keeping scalar fields and
/// the `names` table while dropping parameter/module subgraphs.
fn module_state_shallow(module: &Object) -> Object {
    let Some(state) = module_state_entries(module) else {
        return Object::None;
    };
    let keep = ["nc", "names", "stride", "inplace", "training"];
    let pairs: Vec<(Object, Object)> = state
        .iter()
        .filter(|(k, _)| matches!(k, Object::Unicode(s) if keep.contains(&s.as_str())))
        .filter_map(|(k, v)| shallow_value(v).map(|sv| (k.clone(), sv)))
        .collect();
    Object::Dict(pairs)
}
