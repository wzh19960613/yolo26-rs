//! Slice window generation and per-slice inference orchestration.

use crate::{FilterOption, Image, Result};

use super::merge::merge_detections;
use super::options::{Options, SliceWindow};

use crate::detect::{Model, Prediction};

/// Runs a detector over generated slices, then merges the shifted detections.
pub fn sliced_predict(
    detector: &Model,
    image: &Image,
    inference: &FilterOption,
    options: &Options,
) -> Result<Vec<Prediction>> {
    validate_options(options)?;

    let mut detections = Vec::new();
    if options.include_full_image {
        detections.extend(detector.predict(image, inference)?);
    }

    for window in generate_slices(image.width, image.height, options) {
        let crop = image.crop(window.x, window.y, window.width, window.height)?;
        let (dx, dy) = (window.x as f32, window.y as f32);
        let shifted = detector
            .predict(&crop, inference)?
            .into_iter()
            .map(|d| d.translated(dx, dy));
        detections.extend(shifted);
    }

    Ok(merge_detections(detections, options))
}

/// Generates slice windows that cover the full image.
pub fn generate_slices(image_width: u32, image_height: u32, options: &Options) -> Vec<SliceWindow> {
    let xs = axis_starts(
        image_width,
        options.slice_width.min(image_width),
        options.overlap_width_ratio,
    );
    let ys = axis_starts(
        image_height,
        options.slice_height.min(image_height),
        options.overlap_height_ratio,
    );

    let mut windows = Vec::with_capacity(xs.len() * ys.len());
    for y in ys {
        for &x in &xs {
            windows.push(SliceWindow {
                x,
                y,
                width: options.slice_width.min(image_width - x),
                height: options.slice_height.min(image_height - y),
            });
        }
    }
    windows
}

pub(super) fn validate_options(options: &Options) -> Result<()> {
    if options.slice_width == 0 || options.slice_height == 0 {
        return Err(crate::Error::InvalidConfig(
            "SAHI slice dimensions must be greater than zero".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&options.overlap_width_ratio)
        || !(0.0..1.0).contains(&options.overlap_height_ratio)
    {
        return Err(crate::Error::InvalidConfig(
            "SAHI overlap ratios must be in [0, 1)".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&options.match_threshold) {
        return Err(crate::Error::InvalidConfig(
            "SAHI match threshold must be in [0, 1]".to_string(),
        ));
    }
    Ok(())
}

fn axis_starts(image_len: u32, slice_len: u32, overlap: f32) -> Vec<u32> {
    if image_len <= slice_len {
        return vec![0];
    }

    let step = ((slice_len as f32) * (1.0 - overlap)).round().max(1.0) as u32;
    let last = image_len - slice_len;
    let n = last.div_ceil(step);

    let mut starts: Vec<u32> = (0..n).map(|i| i * step).collect();
    if starts.last().copied() != Some(last) {
        starts.push(last);
    }
    starts
}
