//! Training session: holds a trainable model and optimizer, drives one batch or
//! a full dataset loop.

pub(crate) mod batch;
pub(crate) mod dataset_loop;
pub(crate) mod ema;
pub(crate) mod epoch;
pub(crate) mod loop_steps;
pub(crate) mod methods;
pub(crate) mod optimizer_state;

use crate::model::ImageSize;
pub(crate) use crate::train::exports::*;
use crate::train::model::Model;
use crate::train::optimizer::state::OptimizerState;

/// Minimal train session around a `Model` and optimizer.
pub struct Session {
    pub(crate) model: Model,
    pub(crate) optimizer: OptimizerState,
    /// Optional model-weight EMA shadow, maintained by the training loop and
    /// saved as the best/last checkpoint when active.
    pub(crate) ema: Option<ema::ModelEma>,
}

pub(crate) use batch::*;
pub(crate) use dataset_loop::*;
pub(crate) use epoch::*;
pub(crate) use loop_steps::*;
pub use methods::*;
pub(crate) use optimizer_state::*;
