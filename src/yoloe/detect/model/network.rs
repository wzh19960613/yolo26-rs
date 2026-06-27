use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::yoloe::detect::head::prompt_free::OfficialDetect;
use crate::yoloe::detect::head::{Head, HeadParts};
use crate::yoloe::detect::model::config::Config;
use crate::yoloe::head::lrpc::output::OfficialPyramidOutput;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::prompt::session::Session;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::savpe::encoder_load::load_savpe_gated;
use crate::yoloe::usage::Usage;

pub(crate) struct Network {
    pub(crate) backbone: crate::network::backbone::Base,
    pub(crate) neck: crate::network::neck::Base,
    pub(crate) head: Option<Head>,
    pub(crate) lrpc: Option<Pyramid>,
    pub(crate) prompt_free_head: Option<OfficialDetect>,
    pub(crate) savpe: Option<Encoder>,
}

impl Network {
    pub(crate) fn load(vb: VarBuilder, config: &Config) -> crate::Result<Self> {
        let backbone = crate::network::backbone::Base::load(vb.clone(), config.scale)?;
        let neck = crate::network::neck::Base::load(vb.clone(), config.scale)?;
        let input_channels = config.scale.head_input_channels();
        let cls_hidden = (config.cls_hidden > 0).then_some(config.cls_hidden);
        let box_hidden = (config.box_hidden > 0).then_some(config.box_hidden);
        let head = if config.prompt_head {
            Some(Head::load_with_hidden(
                vb.pp("23"),
                &input_channels,
                config.embed_dim,
                config.max_predictions,
                config.contrastive,
                cls_hidden,
                box_hidden,
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
            Some(OfficialDetect::load(vb.pp("23"), config)?)
        } else {
            None
        };
        let savpe = load_savpe_gated(
            vb.pp("23").pp("savpe"),
            config.official_savpe,
            config.prompt_head,
            config.savpe_hidden,
            config.embed_dim,
            &input_channels,
        )?;
        Ok(Self {
            backbone,
            neck,
            head,
            lrpc,
            prompt_free_head,
            savpe,
        })
    }

    pub(crate) fn pyramid(&self, input: &Tensor) -> crate::Result<[Tensor; 3]> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        Ok([pyramid.small, pyramid.medium, pyramid.large])
    }

    pub(crate) fn forward_parts(
        &self,
        input: &Tensor,
        session: &Session,
    ) -> crate::Result<HeadParts> {
        let pyramid = self.pyramid(input)?;
        let features = pyramid.iter().collect::<Vec<_>>();
        session.forward_open_vocabulary_parts(self.prompt_head()?, &features)
    }

    pub(crate) fn forward(&self, input: &Tensor, session: &Session) -> crate::Result<Tensor> {
        if session.prompts.active_usage() == Some(Usage::PromptFree) {
            if let Some(prompt_free_head) = &self.prompt_free_head {
                session.ensure_prompt_free_class_count(prompt_free_head.classes())?;
                let pyramid = self.pyramid(input)?;
                let features = pyramid.iter().collect::<Vec<_>>();
                return prompt_free_head.forward(
                    &features,
                    session.predict.lrpc_confidence_threshold,
                    session.predict.agnostic_nms,
                );
            }
            let lrpc = self.lrpc.as_ref().ok_or_else(|| {
                crate::Error::InvalidConfig(
                    "YOLOE prompt-free forward requires official LRPC weights".to_string(),
                )
            })?;
            let pyramid = self.pyramid(input)?;
            let features = pyramid.iter().collect::<Vec<_>>();
            return session.forward_official_lrpc_head(self.prompt_head()?, lrpc, &features);
        }
        let pyramid = self.pyramid(input)?;
        let features = pyramid.iter().collect::<Vec<_>>();
        session.forward_open_vocabulary_head(self.prompt_head()?, &features)
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
    ) -> crate::Result<OfficialPyramidOutput> {
        if session.prompts.active_usage() != Some(Usage::PromptFree) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt-free parts require active prompt-free vocabulary".to_string(),
            ));
        }
        if let Some(prompt_free_head) = &self.prompt_free_head {
            session.ensure_prompt_free_class_count(prompt_free_head.classes())?;
            let pyramid = self.pyramid(input)?;
            let features = pyramid.iter().collect::<Vec<_>>();
            return prompt_free_head
                .forward_parts(&features, session.predict.lrpc_confidence_threshold);
        }
        let lrpc = self.lrpc.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig(
                "YOLOE prompt-free parts require official LRPC weights".to_string(),
            )
        })?;
        session.ensure_prompt_free_class_count(lrpc.classes())?;
        let pyramid = self.pyramid(input)?;
        let features = pyramid.iter().collect::<Vec<_>>();
        self.prompt_head()?.forward_official_lrpc_parts(
            &features,
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
