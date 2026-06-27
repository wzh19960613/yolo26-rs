//! Train/eval raw forward dispatch for [`Model`].
//!
//! Extracted from [`crate::train::model::methods`]: the per-task
//! `forward_raw_inner` switches between the dense-prediction tasks'
//! e2e-training / eval heads and the classification / semantic single-output
//! paths, keeping the large 6-arm match off the methods file.

use super::*;

impl Model {
    /// Runs the train-time raw forward path.
    pub fn forward_raw(&self, input: &Tensor) -> crate::Result<Output> {
        crate::network::blocks::with_training_mode(true, || self.forward_raw_inner(input, true))
    }

    /// Runs the eval-time raw forward path.
    pub fn forward_raw_eval(&self, input: &Tensor) -> crate::Result<Output> {
        crate::network::blocks::with_training_mode(false, || self.forward_raw_inner(input, false))
    }

    /// Runs the eval-time inference head used by official end-to-end validation.
    ///
    /// Raw outputs are still needed for validation loss, but end-to-end
    /// detection metrics are computed from the already postprocessed top-k head
    /// output (`[x1, y1, x2, y2, score, class, ...]`).
    pub(crate) fn forward_eval_postprocess(
        &self,
        input: &Tensor,
    ) -> crate::Result<Option<EvalPostprocessOutput>> {
        crate::network::blocks::with_training_mode(false, || match &self.network {
            TrainableNetwork::Detect(net) => {
                let pyramid = net.forward_pyramid(input)?;
                let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
                Ok(Some(EvalPostprocessOutput::Detect {
                    predictions: net.head.forward(&features)?,
                }))
            }
            TrainableNetwork::Segment(net) => {
                let pyramid = net.forward_pyramid(input)?;
                let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
                let (predictions, proto) = net.head.forward(&features)?;
                Ok(Some(EvalPostprocessOutput::Segment { predictions, proto }))
            }
            _ => Ok(None),
        })
    }

    fn forward_raw_inner(&self, input: &Tensor, training: bool) -> crate::Result<Output> {
        match &self.network {
            TrainableNetwork::Detect(net) => self.forward_detect(net, input, training),
            TrainableNetwork::Classify(net) => Ok(Output::Classify {
                logits: net.forward_logits(input)?,
            }),
            TrainableNetwork::Segment(net) => self.forward_segment(net, input, training),
            TrainableNetwork::Pose(net) => self.forward_pose(net, input, training),
            TrainableNetwork::Semantic(net) => Ok(Output::Semantic {
                logits: net.forward(input)?,
            }),
            TrainableNetwork::Obb(net) => self.forward_obb(net, input, training),
        }
    }

    fn forward_detect(
        &self,
        net: &crate::detect::network::Network,
        input: &Tensor,
        training: bool,
    ) -> crate::Result<Output> {
        let pyramid = net.forward_pyramid(input)?;
        let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if training {
            let out = net.head.forward_e2e_training(&features)?;
            Ok(Output::DetectE2e {
                one_to_many: DenseDetectionOutput::from_parts(out.one_to_many),
                one_to_one: DenseDetectionOutput::from_parts(out.one_to_one),
            })
        } else {
            Ok(Output::Detect(DenseDetectionOutput::from_parts(
                net.head.forward_training(&features)?,
            )))
        }
    }

    fn forward_segment(
        &self,
        net: &crate::segment::network::Network,
        input: &Tensor,
        training: bool,
    ) -> crate::Result<Output> {
        let pyramid = net.forward_pyramid(input)?;
        let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if training {
            let out = net.head.forward_e2e_training(&features)?;
            Ok(Output::SegmentE2e {
                one_to_many_detect: DenseDetectionOutput::from_parts(out.one_to_many_detect),
                one_to_many_masks: out.one_to_many_masks,
                one_to_one_detect: DenseDetectionOutput::from_parts(out.one_to_one_detect),
                one_to_one_masks: out.one_to_one_masks,
                proto: out.proto,
                semantic: out.semantic,
            })
        } else {
            let out = net.head.forward_training(&features)?;
            Ok(Output::Segment {
                detect: DenseDetectionOutput::from_parts(out.detect),
                masks: out.masks,
                proto: out.proto,
                semantic: out.semantic,
            })
        }
    }

    fn forward_pose(
        &self,
        net: &crate::pose::network::Network,
        input: &Tensor,
        training: bool,
    ) -> crate::Result<Output> {
        let pyramid = net.forward_pyramid(input)?;
        let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if training {
            let out = net.head.forward_e2e_training(&features)?;
            Ok(Output::PoseE2e {
                one_to_many_detect: DenseDetectionOutput::from_parts(out.one_to_many_detect),
                one_to_many_keypoints: out.one_to_many_keypoints,
                one_to_one_detect: DenseDetectionOutput::from_parts(out.one_to_one_detect),
                one_to_one_keypoints: out.one_to_one_keypoints,
            })
        } else {
            let out = net.head.forward_training(&features)?;
            Ok(Output::Pose {
                detect: DenseDetectionOutput::from_parts(out.detect),
                keypoints: out.keypoints,
            })
        }
    }

    fn forward_obb(
        &self,
        net: &crate::obb::network::Network,
        input: &Tensor,
        training: bool,
    ) -> crate::Result<Output> {
        let pyramid = net.forward_pyramid(input)?;
        let features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if training {
            let out = net.head.forward_e2e_training(&features)?;
            Ok(Output::ObbE2e {
                one_to_many_detect: DenseDetectionOutput::from_parts(out.one_to_many_detect),
                one_to_many_angles: out.one_to_many_angles,
                one_to_one_detect: DenseDetectionOutput::from_parts(out.one_to_one_detect),
                one_to_one_angles: out.one_to_one_angles,
            })
        } else {
            let out = net.head.forward_training(&features)?;
            Ok(Output::Obb {
                detect: DenseDetectionOutput::from_parts(out.detect),
                angles: out.angles,
            })
        }
    }
}
