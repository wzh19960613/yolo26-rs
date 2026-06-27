use candle_core::Tensor;

use crate::network::head::anchors::make_anchors;

use crate::yoloe::detect::head::{Head, HeadParts};
use crate::yoloe::usage::EmbeddingTable;

impl Head {
    /// Runs the head and returns raw box distances, prompt logits, and embeddings.
    pub fn forward_parts(
        &self,
        features: &[&Tensor],
        prompts: &EmbeddingTable,
    ) -> crate::Result<HeadParts> {
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
        let mut feat_sizes = Vec::with_capacity(features.len());

        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            let spatial = h * w;
            feat_sizes.push((h, w));
            all_boxes.push(self.box_branches[i].forward(feature, batch, spatial)?);
            let embedding_map = self.embedding_branches[i].forward_map(feature)?;
            let scores = match self.bn_contrastive_heads.as_ref() {
                Some(heads) => heads[i].forward(&embedding_map, prompts)?,
                None => self.contrastive.forward(&embedding_map, prompts)?,
            }
            .reshape((batch, prompts.class_count(), spatial))?;
            let embeddings = embedding_map.reshape((batch, self.embed_dim, spatial))?;
            all_scores.push(scores);
            all_embeddings.push(embeddings);
        }

        let boxes = Tensor::cat(&all_boxes.iter().collect::<Vec<_>>(), 2)?;
        let scores = Tensor::cat(&all_scores.iter().collect::<Vec<_>>(), 2)?;
        let embeddings = Tensor::cat(&all_embeddings.iter().collect::<Vec<_>>(), 2)?;
        let (anchors, stride_tensor) = make_anchors(&feat_sizes, &self.strides, dtype, device)?;
        Ok(HeadParts {
            boxes,
            scores,
            embeddings,
            anchors,
            stride_tensor,
        })
    }
}
