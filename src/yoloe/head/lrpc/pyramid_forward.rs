use candle_core::Tensor;

use crate::network::head::anchors::make_anchors;

use crate::yoloe::head::lrpc::output::OfficialPyramidOutput;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::select_lrpc_indices::{pad_last_dim, pad_last_dim_with};

impl Pyramid {
    /// Runs official-style LRPC over the three prediction scales.
    pub fn forward(
        &self,
        cls_features: &[&Tensor],
        loc_features: &[&Tensor],
        confidence_threshold: f32,
    ) -> crate::Result<OfficialPyramidOutput> {
        if cls_features.len() != self.heads.len() || loc_features.len() != self.heads.len() {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC pyramid expects {} class and loc feature maps, got {} and {}",
                self.heads.len(),
                cls_features.len(),
                loc_features.len()
            )));
        }
        let batch = cls_features[0].dim(0)?;
        let device = cls_features[0].device();
        let dtype = cls_features[0].dtype();
        let mut feat_sizes = Vec::with_capacity(self.heads.len());
        let mut offsets = Vec::with_capacity(self.heads.len());
        let mut offset = 0usize;
        let mut outputs = Vec::with_capacity(self.heads.len());
        for i in 0..self.heads.len() {
            let (_, _, h, w) = cls_features[i].dims4()?;
            if cls_features[i].dim(0)? != batch || loc_features[i].dim(0)? != batch {
                return Err(crate::Error::InvalidTensor(
                    "YOLOE LRPC pyramid feature maps must have a consistent batch size".to_string(),
                ));
            }
            feat_sizes.push((h, w));
            offsets.push(offset);
            offset += h * w;
            outputs.push(self.heads[i].forward(
                cls_features[i],
                loc_features[i],
                confidence_threshold,
            )?);
        }
        let (anchors, stride_tensor) = make_anchors(&feat_sizes, &self.strides, dtype, device)?;
        let anchors = anchors.squeeze(0)?;
        let stride_tensor = stride_tensor.reshape((1, offset))?;

        let mut batch_boxes = Vec::with_capacity(batch);
        let mut batch_scores = Vec::with_capacity(batch);
        let mut batch_anchors = Vec::with_capacity(batch);
        let mut batch_strides = Vec::with_capacity(batch);
        let mut selected_indices = Vec::with_capacity(batch);
        let mut max_selected = 0usize;

        for b in 0..batch {
            let mut box_parts = Vec::with_capacity(self.heads.len());
            let mut score_parts = Vec::with_capacity(self.heads.len());
            let mut anchor_parts = Vec::with_capacity(self.heads.len());
            let mut stride_parts = Vec::with_capacity(self.heads.len());
            let mut global_indices = Vec::new();
            for (i, output) in outputs.iter().enumerate() {
                let local_indices = &output.selected_indices[b];
                if local_indices.is_empty() {
                    continue;
                }
                let local_index_tensor = Tensor::new(
                    local_indices
                        .iter()
                        .map(|idx| *idx as u32)
                        .collect::<Vec<_>>(),
                    device,
                )?;
                let global = local_indices
                    .iter()
                    .map(|idx| offsets[i] + *idx)
                    .collect::<Vec<_>>();
                let global_index_tensor = Tensor::new(
                    global.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
                    device,
                )?;
                let (_, _, h, w) = cls_features[i].dims4()?;
                let spatial = h * w;
                box_parts.push(
                    output
                        .boxes
                        .narrow(0, b, 1)?
                        .reshape((self.box_channels, spatial))?
                        .contiguous()?
                        .index_select(&local_index_tensor, 1)?,
                );
                score_parts.push(output.scores.narrow(0, b, 1)?.squeeze(0)?.narrow(
                    1,
                    0,
                    local_indices.len(),
                )?);
                anchor_parts.push(
                    anchors
                        .contiguous()?
                        .index_select(&global_index_tensor, 1)?,
                );
                stride_parts.push(
                    stride_tensor
                        .contiguous()?
                        .index_select(&global_index_tensor, 1)?,
                );
                global_indices.extend(global);
            }
            let selected = global_indices.len();
            max_selected = max_selected.max(selected);
            selected_indices.push(global_indices);
            batch_boxes.push(Tensor::cat(&box_parts.iter().collect::<Vec<_>>(), 1)?);
            batch_scores.push(Tensor::cat(&score_parts.iter().collect::<Vec<_>>(), 1)?);
            batch_anchors.push(Tensor::cat(&anchor_parts.iter().collect::<Vec<_>>(), 1)?);
            batch_strides.push(Tensor::cat(&stride_parts.iter().collect::<Vec<_>>(), 1)?);
        }

        let mut boxes = Vec::with_capacity(batch);
        let mut scores = Vec::with_capacity(batch);
        let mut selected_anchors = Vec::with_capacity(batch);
        let mut selected_strides = Vec::with_capacity(batch);
        let mut valid = Vec::with_capacity(batch);
        for b in 0..batch {
            let selected = selected_indices[b].len();
            boxes.push(
                pad_last_dim(&batch_boxes[b], max_selected)?
                    .unsqueeze(0)?
                    .contiguous()?,
            );
            scores.push(
                pad_last_dim_with(&batch_scores[b], max_selected, -1.0e4)?
                    .unsqueeze(0)?
                    .contiguous()?,
            );
            selected_anchors.push(
                pad_last_dim(&batch_anchors[b], max_selected)?
                    .unsqueeze(0)?
                    .contiguous()?,
            );
            selected_strides.push(
                pad_last_dim(&batch_strides[b], max_selected)?
                    .unsqueeze(0)?
                    .contiguous()?,
            );
            let mut valid_b = vec![1.0f32; selected];
            valid_b.resize(max_selected, 0.0);
            valid.push(Tensor::new(valid_b, device)?.unsqueeze(0)?);
        }

        Ok(OfficialPyramidOutput {
            boxes: Tensor::cat(&boxes.iter().collect::<Vec<_>>(), 0)?,
            scores: Tensor::cat(&scores.iter().collect::<Vec<_>>(), 0)?,
            valid: Tensor::cat(&valid.iter().collect::<Vec<_>>(), 0)?,
            anchors: Tensor::cat(&selected_anchors.iter().collect::<Vec<_>>(), 0)?,
            stride_tensor: Tensor::cat(&selected_strides.iter().collect::<Vec<_>>(), 0)?,
            selected_indices,
        })
    }
}
