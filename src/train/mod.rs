//! Native Candle training foundation.
//!
//! This module provides the trainable parameter path, task-neutral raw outputs,
//! checkpoint load/save helpers, a dataset abstraction, and a small training
//! session wrapper. Classification, detection, instance segmentation, pose, and
//! semantic segmentation have supervised training steps with component-level
//! loss reporting for Ultralytics-style log and regression comparisons.

// Training keeps a crate-internal prelude so task-specific submodules can share
// numeric loss/eval helpers across feature combinations. Some imports are only
// consumed by specific task paths, so `cargo check --features train,yoloe` can
// otherwise report false-positive unused import/dead-code noise.
#![allow(ambiguous_glob_reexports, dead_code, unused_imports)]

mod atan_op;
pub mod augment;
mod best_metric;
pub mod checkpoint;
mod class_filter;
pub mod dataset;
mod early_stopping;
mod exports;
mod freeze;
mod load_report;
mod lr_schedule;
/// Trainable task models and per-task network construction.
pub mod model;
/// Optimizer configurations (SGD, AdamW, MuSGD) and state management.
pub mod optimizer;
/// Dataset training-loop runner: epoch loop, validation, reports.
pub mod runner;
/// Training sessions wrapping a model + optimizer for batch and dataset loops.
pub mod session;
mod warmup_schedule;
/// YOLOE-specific training session for fine-tuning segment models.
pub mod yoloe;

// Internal subsystems (loss computation and evaluation metrics). These are
// crate-internal implementation detail; their public types are re-exported at
// `train::` via `exports`.
pub(crate) mod eval;
pub(crate) mod loss;

pub use exports::*;
