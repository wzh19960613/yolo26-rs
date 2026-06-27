use super::*;

pub(crate) fn read_yolo_obb_labels(path: &Path) -> crate::Result<Vec<YoloObbLabel>> {
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
        if cols.len() != 9 {
            return Err(crate::Error::InvalidConfig(format!(
                "invalid OBB label row {} in {}: expected class plus 4 xy corner points",
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
        labels.push(YoloObbLabel {
            class_id,
            points: [
                (parse_f32(1, "x1")?, parse_f32(2, "y1")?),
                (parse_f32(3, "x2")?, parse_f32(4, "y2")?),
                (parse_f32(5, "x3")?, parse_f32(6, "y3")?),
                (parse_f32(7, "x4")?, parse_f32(8, "y4")?),
            ],
        });
    }
    Ok(labels)
}

pub(crate) fn polygon_bounds(points: &[(f32, f32)]) -> Option<(f32, f32, f32, f32)> {
    if points.len() < 3 {
        return None;
    }
    let mut x1 = f32::INFINITY;
    let mut y1 = f32::INFINITY;
    let mut x2 = f32::NEG_INFINITY;
    let mut y2 = f32::NEG_INFINITY;
    for &(x, y) in points {
        x1 = x1.min(x);
        y1 = y1.min(y);
        x2 = x2.max(x);
        y2 = y2.max(y);
    }
    (x2 > x1 && y2 > y1).then_some((x1, y1, x2, y2))
}

pub(crate) fn rasterize_polygon(
    points: &[(f32, f32)],
    width: usize,
    height: usize,
    mask: &mut [f32],
) {
    if points.len() < 3 || mask.len() != width * height {
        return;
    }
    let Some((x1, y1, x2, y2)) = polygon_bounds(points) else {
        return;
    };
    let min_x = x1.floor().max(0.0) as usize;
    let min_y = y1.floor().max(0.0) as usize;
    let max_x = x2.ceil().min(width as f32) as usize;
    let max_y = y2.ceil().min(height as f32) as usize;
    for y in min_y..max_y {
        for x in min_x..max_x {
            if point_in_polygon(x as f32 + 0.5, y as f32 + 0.5, points) {
                mask[y * width + x] = 1.0;
            }
        }
    }
}

fn point_in_polygon(x: f32, y: f32, points: &[(f32, f32)]) -> bool {
    let mut inside = false;
    let mut prev = points.len() - 1;
    for current in 0..points.len() {
        let (xi, yi) = points[current];
        let (xj, yj) = points[prev];
        let denom = yj - yi;
        let intersects = (yi > y) != (yj > y)
            && denom.abs() > f32::EPSILON
            && x < (xj - xi) * (y - yi) / denom + xi;
        if intersects {
            inside = !inside;
        }
        prev = current;
    }
    inside
}

pub(crate) fn label_path_for_image(image_path: &Path) -> PathBuf {
    let mut replaced = PathBuf::new();
    let mut did_replace = false;
    for component in image_path.components() {
        match component {
            Component::Normal(part) if part == "images" && !did_replace => {
                replaced.push("labels");
                did_replace = true;
            }
            other => replaced.push(other.as_os_str()),
        }
    }
    let mut label = if did_replace {
        replaced
    } else {
        image_path.to_path_buf()
    };
    label.set_extension("txt");
    label
}

pub(crate) fn semantic_mask_path_for_image(image_path: &Path) -> PathBuf {
    let mut replaced = PathBuf::new();
    let mut did_replace = false;
    for component in image_path.components() {
        match component {
            Component::Normal(part) if part == "images" && !did_replace => {
                replaced.push("masks");
                did_replace = true;
            }
            other => replaced.push(other.as_os_str()),
        }
    }
    let mut mask = if did_replace {
        replaced
    } else {
        image_path.to_path_buf()
    };
    mask.set_extension("png");
    mask
}
