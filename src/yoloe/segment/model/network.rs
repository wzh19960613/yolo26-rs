use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::yoloe::head::lrpc::output::OfficialSegmentParts;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::segment::head::lrpc::OfficialSegment;
use crate::yoloe::segment::head::{Head, HeadParts};
use crate::yoloe::segment::model::config::Config;
use crate::yoloe::usage::Usage;

pub(crate) struct Network {
    pub(crate) backbone: crate::network::backbone::Base,
    pub(crate) neck: crate::network::neck::Base,
    pub(crate) head: Option<Head>,
    pub(crate) lrpc: Option<Pyramid>,
    pub(crate) prompt_free_head: Option<OfficialSegment>,
    pub(crate) savpe: Option<Encoder>,
}

impl Network {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> crate::Result<Self> {
        let backbone = crate::network::backbone::Base::load(vb.clone(), config.scale)?;
        let neck = crate::network::neck::Base::load(vb.clone(), config.scale)?;
        let input_channels = config.scale.head_input_channels();
        // Use checkpoint-inferred hidden widths when available so the built head
        // matches the official layout exactly (None falls back to formulas).
        let cls_hidden = (config.cls_hidden > 0).then_some(config.cls_hidden);
        let box_hidden = (config.box_hidden > 0).then_some(config.box_hidden);
        let mask_hidden = (config.mask_hidden > 0).then_some(config.mask_hidden);
        let head = if config.prompt_head {
            Some(Head::load_with_hidden(
                vb.pp("23"),
                &input_channels,
                config.embed_dim,
                config.max_predictions,
                config.mask_channels,
                config.proto_channels,
                config.contrastive,
                config.mask_branch.as_str(),
                cls_hidden,
                box_hidden,
                mask_hidden,
                true,
            )?)
        } else {
            None
        };
        let lrpc = if config.official_lrpc && config.prompt_head {
            Some(Pyramid::load_inferred(
                vb.pp("23").pp("lrpc"),
                config.embed_dim,
                4,
                true,
            )?)
        } else {
            None
        };
        let prompt_free_head = if config.official_lrpc && !config.prompt_head {
            Some(OfficialSegment::load(vb.pp("23"), config)?)
        } else {
            None
        };
        let savpe = Self::load_savpe(vb.pp("23").pp("savpe"), config, &input_channels)?;
        Ok(Self {
            backbone,
            neck,
            head,
            lrpc,
            prompt_free_head,
            savpe,
        })
    }

    pub(crate) fn forward_parts(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<HeadParts> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        let head = self.prompt_head()?;
        session.forward_open_vocabulary_segment_parts(head, &head_features)
    }

    pub(crate) fn forward(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<(Tensor, Tensor)> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if session.prompts.active_usage() == Some(Usage::PromptFree) {
            if let Some(prompt_free_head) = &self.prompt_free_head {
                session.ensure_prompt_free_class_count(prompt_free_head.classes())?;
                return prompt_free_head.forward(
                    &head_features,
                    session.predict.lrpc_confidence_threshold,
                    session.predict.agnostic_nms,
                );
            }
            let lrpc = self.lrpc.as_ref().ok_or_else(|| {
                crate::Error::InvalidConfig(
                    "YOLOE prompt-free forward requires official LRPC weights".to_string(),
                )
            })?;
            let head = self.prompt_head()?;
            return session.forward_official_lrpc_segment_head(head, lrpc, &head_features);
        }
        let head = self.prompt_head()?;
        session.forward_open_vocabulary_segment_head(head, &head_features)
    }

    pub(crate) fn prompt_free_class_count(&self) -> Option<usize> {
        self.prompt_free_head
            .as_ref()
            .map(|head| head.classes())
            .or_else(|| self.lrpc.as_ref().map(|lrpc| lrpc.classes()))
    }

    pub(crate) fn forward_prompt_free_parts(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<OfficialSegmentParts> {
        if session.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt-free parts require active prompt-free vocabulary".to_string(),
            ));
        }
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        if let Some(prompt_free_head) = &self.prompt_free_head {
            session.ensure_prompt_free_class_count(prompt_free_head.classes())?;
            return prompt_free_head
                .forward_parts(&head_features, session.predict.lrpc_confidence_threshold);
        }
        let lrpc = self.lrpc.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE prompt-free parts require official LRPC weights".to_string(),
            )
        })?;
        session.ensure_prompt_free_class_count(lrpc.classes())?;
        self.prompt_head()?.forward_official_lrpc_parts(
            &head_features,
            lrpc,
            session.predict.lrpc_confidence_threshold,
        )
    }

    pub(crate) fn prompt_head(&self) -> crate::Result<&Head> {
        self.head.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE text/visual prompt forward requires a checkpoint with prompt head weights"
                    .to_string(),
            )
        })
    }
}
