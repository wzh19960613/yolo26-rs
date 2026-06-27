use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::network::blocks::ConvBlock;

use crate::yoloe::savpe::encoder::{Encoder, SavpeFeaturePath};

impl Encoder {
    /// Loads official-style `savpe.*` weights with explicit hidden and embedding dimensions.
    pub fn load(
        vb: VarBuilder,
        input_channels: &[usize],
        hidden_channels: usize,
        embed_dim: usize,
    ) -> crate::Result<Self> {
        if input_channels.len() != 3 {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE SAVPE expects 3 feature scales, got {}",
                input_channels.len()
            )));
        }
        if hidden_channels == 0 || embed_dim == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE SAVPE hidden_channels and embed_dim must be greater than zero".to_string(),
            ));
        }
        let attention_channels = 16;
        if !embed_dim.is_multiple_of(attention_channels) {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE SAVPE embed_dim {embed_dim} must be divisible by attention channels {attention_channels}"
            )));
        }
        let mut cv1 = Vec::with_capacity(input_channels.len());
        let mut cv2 = Vec::with_capacity(input_channels.len());
        for (i, &channels) in input_channels.iter().enumerate() {
            let scale = match i {
                0 => 1,
                1 => 2,
                2 => 4,
                other => {
                    return Err(crate::Error::InvalidConfig(format!(
                        "YOLOE SAVPE feature scale index out of range: {other} (expected 0..3)"
                    )));
                }
            };
            cv1.push(SavpeFeaturePath::two_conv(
                vb.pp("cv1").pp(i.to_string()),
                channels,
                hidden_channels,
                scale,
            )?);
            cv2.push(SavpeFeaturePath::one_conv(
                vb.pp("cv2").pp(i.to_string()),
                channels,
                hidden_channels,
                scale,
            )?);
        }
        let cfg1 = Conv2dConfig::default();
        let cfg3 = Conv2dConfig {
            padding: 1,
            ..Conv2dConfig::default()
        };
        Ok(Self {
            cv1,
            cv2,
            cv3: conv2d(3 * hidden_channels, embed_dim, 1, cfg1, vb.pp("cv3"))?,
            cv4: conv2d(
                3 * hidden_channels,
                attention_channels,
                3,
                cfg3,
                vb.pp("cv4"),
            )?,
            cv5: conv2d(1, attention_channels, 3, cfg3, vb.pp("cv5"))?,
            cv6_0: ConvBlock::load(
                vb.pp("cv6").pp("0"),
                2 * attention_channels,
                attention_channels,
                3,
                1,
                1,
                true,
            )?,
            cv6_1: conv2d(
                attention_channels,
                attention_channels,
                3,
                cfg3,
                vb.pp("cv6").pp("1"),
            )?,
            attention_channels,
            hidden_channels,
            embed_dim,
        })
    }

    /// Loads official-style weights using the Ultralytics hidden-channel rule.
    pub fn load_with_class_count(
        vb: VarBuilder,
        input_channels: &[usize],
        embed_dim: usize,
        class_count: usize,
    ) -> crate::Result<Self> {
        let hidden_channels = input_channels[0].max(class_count.min(100));
        Self::load(vb, input_channels, hidden_channels, embed_dim)
    }

    /// Returns the intermediate feature channel count used by official SAVPE.
    pub const fn hidden_channels(&self) -> usize {
        self.hidden_channels
    }

    /// Returns the output prompt embedding dimension.
    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

/// Loads official SAVPE weights when the checkpoint reports them as available.
///
/// Returns `Ok(None)` when `official_savpe` is false or the prompt head is
/// absent (the prompt-free head variant omits SAVPE). Otherwise loads the
/// encoder from `vb` (expected to be pre-prefixed to the `*.savpe` subkey)
/// using the inferred `savpe_hidden` and `embed_dim`. Model and segment
/// networks share this gate so both can attach visual-prompt encoding.
pub(crate) fn load_savpe_gated(
    vb: VarBuilder,
    official_savpe: bool,
    prompt_head: bool,
    savpe_hidden: usize,
    embed_dim: usize,
    input_channels: &[usize],
) -> crate::Result<Option<Encoder>> {
    if !official_savpe || !prompt_head {
        return Ok(None);
    }
    Ok(Some(Encoder::load(
        vb,
        input_channels,
        savpe_hidden,
        embed_dim,
    )?))
}
