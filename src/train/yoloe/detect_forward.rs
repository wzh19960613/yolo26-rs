use crate::network::head::anchors::make_anchors;

use super::detect_output::HeadOutput;
use crate::yoloe::detect::head::Head;
use crate::yoloe::usage::EmbeddingTable;

use candle_core::Tensor;

impl Head {
    /// Training-time forward: BatchNorm runs in train mode and the per-scale
    /// class/box feature maps are returned so LRPC can reuse them.
    pub fn forward_dense_parts(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<HeadOutput> {
        if features.len() != self.box_branches.len() {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE open-vocabulary head expected {} feature maps, got {}",
                self.box_branches.len(),
                features.len()
            )));
        }
        if prompts.dim()? != self.embed_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE prompt dim {} does not match head embed_dim {}",
                prompts.dim()?,
                self.embed_dim
            )));
        }
        let device = features[0].device();
        let dtype = features[0].dtype();
        let batch = features[0].dim(0)?;
        let mut all_boxes = Vec::with_capacity(features.len());
        let mut all_scores = Vec::with_capacity(features.len());
        let mut all_embeddings = Vec::with_capacity(features.len());
        let mut cls_maps = Vec::with_capacity(features.len());
        let mut loc_maps = Vec::with_capacity(features.len());
        let mut feat_sizes = Vec::with_capacity(features.len());

        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            let spatial = h * w;
            feat_sizes.push((h, w));
            loc_maps.push(self.box_branches[i].forward_map(feature)?);
            all_boxes.push(
                self.box_branches[i]
                    .forward_map(feature)?
                    .reshape((batch, 4, spatial))?,
            );
            let embedding_map = self.embedding_branches[i].forward_map(feature)?;
            let scores = match self.bn_contrastive_heads.as_ref() {
                Some(heads) => heads[i].forward_dense(&embedding_map, prompts)?,
                None => self.contrastive.forward(&embedding_map, prompts)?,
            }
            .reshape((batch, prompts.class_count(), spatial))?;
            let embeddings = embedding_map.reshape((batch, self.embed_dim, spatial))?;
            cls_maps.push(self.embedding_branches[i].forward_map(feature)?);
            all_scores.push(scores);
            all_embeddings.push(embeddings);
        }

        let boxes = Tensor::cat(&all_boxes.iter().collect::<Vec<_>>(), 2)?;
        let scores = Tensor::cat(&all_scores.iter().collect::<Vec<_>>(), 2)?;
        let embeddings = Tensor::cat(&all_embeddings.iter().collect::<Vec<_>>(), 2)?;
        let (anchors, stride_tensor) = make_anchors(&feat_sizes, &self.strides, dtype, device)?;
        Ok(HeadOutput {
            boxes,
            scores,
            embeddings,
            anchors,
            stride_tensor,
            cls_feature_maps: cls_maps,
            loc_feature_maps: loc_maps,
        })
    }
}
