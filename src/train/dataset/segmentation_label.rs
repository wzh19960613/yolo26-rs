use super::*;

pub(crate) struct YoloSegmentationLabel {
    pub(crate) class_id: u32,
    pub(crate) points: Vec<(f32, f32)>,
}

pub(crate) fn read_yolo_segmentation_labels(
    path: &Path,
) -> crate::Result<Vec<YoloSegmentationLabel>> {
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
        if cols.len() < 7 || cols.len().is_multiple_of(2) {
            return Err(crate::Error::InvalidConfig(format!(
                "invalid segmentation label row {} in {}: expected class plus at least 3 xy points",
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
        let mut points = Vec::with_capacity((cols.len() - 1) / 2);
        for pair in cols[1..].chunks_exact(2) {
            let x = pair[0].parse::<f32>().map_err(|err| {
                crate::Error::InvalidConfig(format!(
                    "invalid polygon x on row {} in {}: {err}",
                    line_idx + 1,
                    path.display()
                ))
            })?;
            let y = pair[1].parse::<f32>().map_err(|err| {
                crate::Error::InvalidConfig(format!(
                    "invalid polygon y on row {} in {}: {err}",
                    line_idx + 1,
                    path.display()
                ))
            })?;
            points.push((x, y));
        }
        labels.push(YoloSegmentationLabel { class_id, points });
    }
    Ok(labels)
}

#[derive(Debug, Clone, Copy)]

pub(crate) struct YoloPoseKeypoint {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) visibility: f32,
}

#[derive(Debug, Clone)]

pub(crate) struct YoloPoseLabel {
    pub(crate) class_id: u32,
    pub(crate) cx: f32,
    pub(crate) cy: f32,
    pub(crate) w: f32,
    pub(crate) h: f32,
    pub(crate) keypoints: Vec<YoloPoseKeypoint>,
}

pub(crate) fn read_yolo_pose_labels(
    path: &Path,
    keypoints_count: usize,
    keypoint_dims: usize,
) -> crate::Result<Vec<YoloPoseLabel>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    let expected_cols = 5 + keypoints_count * keypoint_dims;
    let mut labels = Vec::new();
    for (line_idx, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() != expected_cols {
            return Err(crate::Error::InvalidConfig(format!(
                "invalid pose label row {} in {}: expected {expected_cols} columns for kpt_shape [{keypoints_count}, {keypoint_dims}], got {}",
                line_idx + 1,
                path.display(),
                cols.len()
            )));
        }
        let parse_f32 = |idx: usize, name: &str| -> crate::Result<f32> {
            cols[idx].parse::<f32>().map_err(|err| {
                crate::Error::InvalidConfig(format!(
                    "invalid {name} on row {} in {}: {err}",
                    line_idx + 1,
                    path.display()
                ))
            })
        };
        let class_id = cols[0].parse::<u32>().map_err(|err| {
            crate::Error::InvalidConfig(format!(
                "invalid class id on row {} in {}: {err}",
                line_idx + 1,
                path.display()
            ))
        })?;
        let mut keypoints = Vec::with_capacity(keypoints_count);
        for keypoint_idx in 0..keypoints_count {
            let base = 5 + keypoint_idx * keypoint_dims;
            let x = parse_f32(base, "keypoint x")?;
            let y = parse_f32(base + 1, "keypoint y")?;
            let visibility = if keypoint_dims >= 3 {
                parse_f32(base + 2, "keypoint visibility")?
            } else if x > 0.0 || y > 0.0 {
                1.0
            } else {
                0.0
            };
            keypoints.push(YoloPoseKeypoint {
                x,
                y,
                visibility: if visibility > 0.0 { 1.0 } else { 0.0 },
            });
        }
        labels.push(YoloPoseLabel {
            class_id,
            cx: parse_f32(1, "cx")?,
            cy: parse_f32(2, "cy")?,
            w: parse_f32(3, "w")?,
            h: parse_f32(4, "h")?,
            keypoints,
        });
    }
    Ok(labels)
}

#[derive(Debug, Clone)]

pub(crate) struct YoloObbLabel {
    pub(crate) class_id: u32,
    pub(crate) points: [(f32, f32); 4],
}
