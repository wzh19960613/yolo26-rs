pub(crate) fn xywhr_from_points(points: &[(f32, f32)]) -> Option<[f32; 5]> {
    if points.len() < 3 {
        return None;
    }
    let mut best = None;
    for i in 0..points.len() {
        let (x1, y1) = points[i];
        let (x2, y2) = points[(i + 1) % points.len()];
        let angle = (y2 - y1).atan2(x2 - x1);
        let rect = rect_for_angle(points, angle)?;
        if best.map(|(_, area)| rect.1 < area).unwrap_or(true) {
            best = Some(rect);
        }
    }
    let (mut rbox, _) = best?;
    if rbox[2] < rbox[3] {
        rbox.swap(2, 3);
        rbox[4] += std::f32::consts::FRAC_PI_2;
    }
    rbox[4] = normalize_theta(rbox[4]);
    Some(rbox)
}

pub(crate) fn rbox_xyxy(rbox: [f32; 5]) -> [f32; 4] {
    let points = xywhr_points(rbox);
    let mut x1 = f32::INFINITY;
    let mut y1 = f32::INFINITY;
    let mut x2 = f32::NEG_INFINITY;
    let mut y2 = f32::NEG_INFINITY;
    for (x, y) in points {
        x1 = x1.min(x);
        y1 = y1.min(y);
        x2 = x2.max(x);
        y2 = y2.max(y);
    }
    [x1, y1, x2, y2]
}

pub(crate) fn probiou(a: [f32; 5], b: [f32; 5]) -> f32 {
    let (a1, b1, c1) = covariance(a);
    let (a2, b2, c2) = covariance(b);
    let eps = 1e-7f32;
    let denom = (a1 + a2) * (b1 + b2) - (c1 + c2).powi(2) + eps;
    let t1 = ((a1 + a2) * (a[1] - b[1]).powi(2) + (b1 + b2) * (a[0] - b[0]).powi(2)) / denom * 0.25;
    let t2 = ((c1 + c2) * (b[0] - a[0]) * (a[1] - b[1]) / denom) * 0.5;
    let det1 = (a1 * b1 - c1.powi(2)).max(0.0);
    let det2 = (a2 * b2 - c2.powi(2)).max(0.0);
    let t3 = ((denom - eps) / (4.0 * (det1 * det2).sqrt() + eps) + eps).ln() * 0.5;
    let bd = (t1 + t2 + t3).clamp(eps, 100.0);
    1.0 - (1.0 - (-bd).exp() + eps).sqrt()
}

fn rect_for_angle(points: &[(f32, f32)], angle: f32) -> Option<([f32; 5], f32)> {
    let (cos, sin) = (angle.cos(), angle.sin());
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &(x, y) in points {
        let rx = x * cos + y * sin;
        let ry = -x * sin + y * cos;
        min_x = min_x.min(rx);
        min_y = min_y.min(ry);
        max_x = max_x.max(rx);
        max_y = max_y.max(ry);
    }
    let (w, h) = (max_x - min_x, max_y - min_y);
    if w <= 0.0 || h <= 0.0 {
        return None;
    }
    let (rcx, rcy) = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);
    let cx = rcx * cos - rcy * sin;
    let cy = rcx * sin + rcy * cos;
    Some(([cx, cy, w, h, angle], w * h))
}

fn xywhr_points([cx, cy, w, h, angle]: [f32; 5]) -> [(f32, f32); 4] {
    let (cos, sin) = (angle.cos(), angle.sin());
    let v1 = (w * 0.5 * cos, w * 0.5 * sin);
    let v2 = (-h * 0.5 * sin, h * 0.5 * cos);
    [
        (cx + v1.0 + v2.0, cy + v1.1 + v2.1),
        (cx + v1.0 - v2.0, cy + v1.1 - v2.1),
        (cx - v1.0 - v2.0, cy - v1.1 - v2.1),
        (cx - v1.0 + v2.0, cy - v1.1 + v2.1),
    ]
}

fn normalize_theta(mut theta: f32) -> f32 {
    while theta >= 3.0 * std::f32::consts::FRAC_PI_4 {
        theta -= std::f32::consts::PI;
    }
    while theta < -std::f32::consts::FRAC_PI_4 {
        theta += std::f32::consts::PI;
    }
    theta
}

fn covariance([_, _, w, h, angle]: [f32; 5]) -> (f32, f32, f32) {
    let (cos, sin) = (angle.cos(), angle.sin());
    let (w2, h2) = (w.powi(2) / 12.0, h.powi(2) / 12.0);
    (
        w2 * cos.powi(2) + h2 * sin.powi(2),
        w2 * sin.powi(2) + h2 * cos.powi(2),
        (w2 - h2) * cos * sin,
    )
}
