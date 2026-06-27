#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// YOLO26 model scale variant.
#[cfg_attr(feature = "wasm", wasm_bindgen(js_name = Scale))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scale {
    /// Nano model.
    N,
    /// Small model.
    S,
    /// Medium model.
    M,
    /// Large model.
    L,
    /// Extra-large model.
    X,
}

impl std::fmt::Display for Scale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::N => "n",
            Self::S => "s",
            Self::M => "m",
            Self::L => "l",
            Self::X => "x",
        };
        f.write_str(s)
    }
}

impl TryFrom<char> for Scale {
    type Error = ();

    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            'n' | 'N' => Ok(Self::N),
            's' | 'S' => Ok(Self::S),
            'm' | 'M' => Ok(Self::M),
            'l' | 'L' => Ok(Self::L),
            'x' | 'X' => Ok(Self::X),
            _ => Err(()),
        }
    }
}

impl TryFrom<&str> for Scale {
    type Error = ();

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.len() {
            1 => Self::try_from(s.chars().next().ok_or(())?),
            _ => Err(()),
        }
    }
}

impl Scale {
    /// Returns the model parameters(depth, width, max_channels) for this scale.
    pub fn params(self) -> (f32, f32, usize) {
        match self {
            Self::N => (0.50, 0.25, 1024),
            Self::S => (0.50, 0.50, 1024),
            Self::M => (0.50, 1.00, 512),
            Self::L => (1.00, 1.00, 512),
            Self::X => (1.00, 1.50, 512),
        }
    }

    /// Returns whether all C3k blocks should use the larger kernel variant.
    pub fn c3k_all(self) -> bool {
        matches!(self, Self::M | Self::L | Self::X)
    }

    /// Scales a YAML channel count for this model size.
    pub fn channel(self, yaml_c: usize) -> usize {
        let (_, width, max_channels) = self.params();
        let raw = yaml_c.min(max_channels) as f32 * width;
        ((raw / 8.0).ceil() as usize) * 8
    }

    /// Scales a YAML repeat count for this model size.
    pub fn repeat(self, yaml_n: usize) -> usize {
        if yaml_n > 1 {
            let (depth, _, _) = self.params();
            (yaml_n as f32 * depth).round().max(1.0) as usize
        } else {
            yaml_n
        }
    }

    /// Returns channel counts expected by dense prediction head inputs.
    pub fn head_input_channels(self) -> [usize; 3] {
        [self.channel(256), self.channel(512), self.channel(1024)]
    }
}
