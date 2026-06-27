//! YOLO detection label parsing.
//!
//! Reads a Ultralytics-style `.txt` label file into normalized
//! `cx,cy,w,h` boxes, deriving the bbox from a segmentation polygon
//! (matching the official `segments2boxes`) when one is present.

use super::*;

/// A normalized YOLO detection label: class id plus `cx,cy,w,h` in [0,1].
#[derive(Debug, Clone, Copy)]

pub(crate) struct YoloDetectionLabel {
    pub(crate) class_id: u32,
    pub(crate) cx: f32,
    pub(crate) cy: f32,
    pub(crate) w: f32,
    pub(crate) h: f32,
}

/// Parses a YOLO `.txt` label file into normalized detection labels.
///
/// Missing files yield an empty vector; rows with a trailing segmentation
/// polygon derive their bbox from the polygon's point extents (official
/// `segments2boxes`).
pub(crate) fn read_yolo_detection_labels(path: &Path) -> crate::Result<Vec<YoloDetectionLabel>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    let mut labels = Vec::new();
    for (line_idx, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() < 5 {
            return Err(crate::Error::InvalidConfig(format!(
                "invalid YOLO label row {} in {}: expected at least 5 columns",
                line_idx + 1,
                path.display()
            )));
        }
        let class_id = cols[0].parse::<u32>().map_err(|err| {
            crate::Error::InvalidConfig(format!(
                "invalid class id on row {} in {}: {err}",
                line_idx + 1,
                path.display()
            ))
        })?;
        let parse_f32 = |idx: usize, name: &str| -> crate::Result<f32> {
            cols[idx].parse::<f32>().map_err(|err| {
                crate::Error::InvalidConfig(format!(
                    "invalid {name} on row {} in {}: {err}",
                    line_idx + 1,
                    path.display()
                ))
            })
        };
        // Match the official `segments2boxes`: when a row carries a segmentation
        // polygon (more than the 5 detection columns), derive the bbox from the
        // polygon point min/max rather than the `cx,cy,w,h` prefix the file
        // writer estimated. This is what the Ultralytics segmentation dataloader
        // does and what the official `v8DetectionLoss` therefore sees as GT.
        let (cx, cy, w, h) = if cols.len() > 5 && cols.len() % 2 == 1 {
            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            let mut k = 5;
            while k + 1 < cols.len() {
                let px = parse_f32(k, "polygon x")?;
                let py = parse_f32(k + 1, "polygon y")?;
                min_x = min_x.min(px);
                min_y = min_y.min(py);
                max_x = max_x.max(px);
                max_y = max_y.max(py);
                k += 2;
            }
            (
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                max_x - min_x,
                max_y - min_y,
            )
        } else {
            (
                parse_f32(1, "cx")?,
                parse_f32(2, "cy")?,
                parse_f32(3, "w")?,
                parse_f32(4, "h")?,
            )
        };
        labels.push(YoloDetectionLabel {
            class_id,
            cx,
            cy,
            w,
            h,
        });
    }
    Ok(labels)
}
