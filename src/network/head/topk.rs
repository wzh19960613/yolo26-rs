use candle_core::{DType, Result, Tensor};

pub fn postprocess_topk(preds: &Tensor, nc: usize, extra: usize, max_det: usize) -> Result<Tensor> {
    postprocess_topk_with_mode(preds, nc, extra, max_det, false)
}

pub fn postprocess_topk_with_mode(
    preds: &Tensor,
    nc: usize,
    extra: usize,
    max_det: usize,
    agnostic: bool,
) -> Result<Tensor> {
    let batch = preds.dim(0)?;
    let n_anchors = preds.dim(1)?;
    let k = max_det.min(n_anchors);
    let boxes = preds.narrow(2, 0, 4)?.to_dtype(DType::F32)?.to_vec3()?;
    let scores = preds.narrow(2, 4, nc)?.to_dtype(DType::F32)?.to_vec3()?;
    let extras = if extra == 0 {
        None
    } else {
        Some(
            preds
                .narrow(2, 4 + nc, extra)?
                .to_dtype(DType::F32)?
                .to_vec3::<f32>()?,
        )
    };

    let mut out = Vec::with_capacity(batch * k * (6 + extra));
    for b in 0..batch {
        let candidates = if agnostic {
            agnostic_candidates(&scores[b], nc, k)
        } else {
            class_aware_candidates(&scores[b], nc, k)
        };
        for candidate in candidates {
            out.extend_from_slice(&candidate.output_box(&boxes[b]));
            if let Some(extras) = &extras {
                out.extend_from_slice(&extras[b][candidate.anchor]);
            }
        }
    }

    Tensor::from_vec(out, (batch, k, 6 + extra), preds.device())
}

#[derive(Clone, Copy)]
struct Candidate {
    anchor: usize,
    class_id: usize,
    score: f32,
}

impl Candidate {
    fn output_box(self, boxes: &[Vec<f32>]) -> [f32; 6] {
        [
            boxes[self.anchor][0],
            boxes[self.anchor][1],
            boxes[self.anchor][2],
            boxes[self.anchor][3],
            self.score,
            self.class_id as f32,
        ]
    }
}

fn agnostic_candidates(scores: &[Vec<f32>], nc: usize, k: usize) -> Vec<Candidate> {
    let mut max_scores = Vec::with_capacity(scores.len());
    let mut classes = Vec::with_capacity(scores.len());
    for anchor_scores in scores {
        let mut best_class = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (class_id, score) in anchor_scores.iter().take(nc).enumerate() {
            if *score > best_score {
                best_class = class_id;
                best_score = *score;
            }
        }
        max_scores.push(best_score);
        classes.push(best_class);
    }
    topk_indices(&max_scores, k)
        .into_iter()
        .map(|anchor| Candidate {
            anchor,
            class_id: classes[anchor],
            score: max_scores[anchor],
        })
        .collect()
}

fn class_aware_candidates(scores: &[Vec<f32>], nc: usize, k: usize) -> Vec<Candidate> {
    let top_anchors = {
        let max_scores = scores
            .iter()
            .map(|anchor_scores| {
                anchor_scores
                    .iter()
                    .take(nc)
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect::<Vec<_>>();
        topk_indices(&max_scores, k)
    };
    let mut candidates = Vec::with_capacity(top_anchors.len() * nc);
    for anchor in top_anchors {
        for (class_id, score) in scores[anchor].iter().take(nc).copied().enumerate() {
            candidates.push(Candidate {
                anchor,
                class_id,
                score,
            });
        }
    }
    topk_candidates(candidates, k)
}

fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }

    let mut scored: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| compare_score_desc(a.1, b.1));
        scored.truncate(k);
    }
    scored.sort_unstable_by(|a, b| compare_score_desc(a.1, b.1));
    scored.into_iter().map(|(idx, _)| idx).collect()
}

fn topk_candidates(mut candidates: Vec<Candidate>, k: usize) -> Vec<Candidate> {
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k, |a, b| compare_score_desc(a.score, b.score));
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(|a, b| compare_score_desc(a.score, b.score));
    candidates
}

fn compare_score_desc(a: f32, b: f32) -> std::cmp::Ordering {
    b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
}
