//! Single-image augmentation applied to a training [`Sample`].
//!
//! HSV jitter is image-only and runs for every task. Affine scale/translation is
//! applied to the spatial tasks (detection, segmentation, pose, OBB and the
//! semantic class map) via [`affine_target`], which keeps each task's target
//! aligned with the affined image; classification has no spatial target and
//! skips the affine branch. Horizontal and vertical flips are dispatched
//! through [`flip_target`], covering every box-bearing task and the semantic
//! class map.

use candle_core::Tensor;

use super::affine::affine_plan;
use super::affine::apply_affine_image;
use super::affine_target::affine_target;
use super::flip_target::flip_target;
use super::hsv::{HsvGains, augment_hsv};
use super::{AugmentConfig, Sample, SeededRng, Target};

/// Mirrors a `[1, C, H, W]` image along its width axis.
pub fn flip_image_horizontal(image: &Tensor) -> crate::Result<Tensor> {
    Ok(image.flip(&[3])?.contiguous()?)
}

/// Mirrors a `[1, C, H, W]` image along its height axis.
pub fn flip_image_vertical(image: &Tensor) -> crate::Result<Tensor> {
    Ok(image.flip(&[2])?.contiguous()?)
}

/// Applies HSV jitter, affine scale/translation (box-bearing tasks) and flips
/// (all tasks) to one sample, drawing each decision deterministically from `rng`.
pub(crate) fn augment_sample(
    mut sample: Sample,
    config: &AugmentConfig,
    geom_rng: &mut SeededRng,
    color_rng: &mut SeededRng,
) -> crate::Result<Sample> {
    if config.is_identity() {
        return Ok(sample);
    }
    let width = sample.input.dim(3)? as f32;
    let height = sample.input.dim(2)? as f32;
    let canvas_h = height as usize;
    let canvas_w = width as usize;

    if (config.scale != 0.0 || config.translate != 0.0)
        && matches!(
            sample.target,
            Target::Detection(_)
                | Target::Segmentation(_)
                | Target::Pose(_)
                | Target::Obb(_)
                | Target::Semantic { .. }
        )
    {
        let plan = affine_plan(geom_rng, canvas_h, canvas_w, config.scale, config.translate);
        if plan.s_w != 1.0 || plan.s_h != 1.0 || plan.dx != 0.0 || plan.dy != 0.0 {
            sample.input = apply_affine_image(&sample.input, plan, canvas_h, canvas_w)?;
            sample.target = affine_target(sample.target, plan, width, height)?;
        }
    }

    if config.hsv_h != 0.0 || config.hsv_s != 0.0 || config.hsv_v != 0.0 {
        let dtype = sample.input.dtype();
        let device = sample.input.device().clone();
        let shape = sample.input.dims();
        let mut pixels = sample
            .input
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        augment_hsv(
            &mut pixels,
            HsvGains {
                hgain: config.hsv_h,
                sgain: config.hsv_s,
                vgain: config.hsv_v,
            },
            color_rng,
        );
        sample.input = Tensor::from_vec(pixels, shape, &device)?.to_dtype(dtype)?;
    }

    let do_ud = config.flipud > 0.0 && geom_rng.bernoulli(config.flipud);
    let do_lr = config.fliplr > 0.0 && geom_rng.bernoulli(config.fliplr);
    if do_lr || do_ud {
        if do_lr {
            sample.input = flip_image_horizontal(&sample.input)?;
        }
        if do_ud {
            sample.input = flip_image_vertical(&sample.input)?;
        }
        sample.target = flip_target(sample.target, do_lr, do_ud, width, height)?;
    }

    Ok(sample)
}
