//! Prompt-free LRPC scoring heads and their outputs.

pub(crate) mod feature_branch;
pub(crate) mod head;
pub(crate) mod official;
pub(crate) mod official_load;
pub(crate) mod output;
pub(crate) mod pyramid;
pub(crate) mod pyramid_forward;
pub(crate) mod pyramid_load;

pub use head::LrpcHead;
pub use official::Official;
pub use output::{LrpcOutput, OfficialOutput, OfficialPyramidOutput};
pub use pyramid::Pyramid;
