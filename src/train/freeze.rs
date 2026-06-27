/// Ultralytics-style train-time layer freezing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Freeze {
    /// Freeze the first N `model.{index}.` layers.
    FirstLayers(usize),
    /// Freeze the specified `model.{index}.` layers.
    Layers(Vec<usize>),
}

impl Freeze {
    /// Creates a freeze rule that freezes the first `count` model layers.
    pub const fn first_layers(count: usize) -> Self {
        Self::FirstLayers(count)
    }

    /// Creates a freeze rule that freezes the specified layer indices.
    pub fn layers(layers: impl Into<Vec<usize>>) -> crate::Result<Self> {
        let layers = normalize_layers(layers.into())?;
        Ok(Self::Layers(layers))
    }

    /// Returns true when a named train variable should be frozen.
    pub fn freezes_variable(&self, name: &str) -> bool {
        is_always_frozen_variable(name)
            || match self {
                Self::FirstLayers(count) => (0..*count).any(|layer| has_layer_prefix(name, layer)),
                Self::Layers(layers) => layers.iter().any(|&layer| has_layer_prefix(name, layer)),
            }
    }

    /// Returns true when a named train variable should remain trainable.
    pub fn allows_variable(&self, name: &str) -> bool {
        !self.freezes_variable(name)
    }

    /// Returns the explicit frozen layer indices, when this rule uses a list.
    pub fn layers_list(&self) -> Option<&[usize]> {
        match self {
            Self::Layers(layers) => Some(layers),
            Self::FirstLayers(_) => None,
        }
    }

    /// Returns the frozen prefix count, when this rule freezes the first N layers.
    pub const fn first_layers_count(&self) -> Option<usize> {
        match self {
            Self::FirstLayers(count) => Some(*count),
            Self::Layers(_) => None,
        }
    }
}

/// Returns true for variables that Ultralytics always freezes.
pub fn is_always_frozen_variable(name: &str) -> bool {
    name.contains(".dfl")
}

fn has_layer_prefix(name: &str, layer: usize) -> bool {
    name.contains(&format!("model.{layer}."))
}

fn normalize_layers(mut layers: Vec<usize>) -> crate::Result<Vec<usize>> {
    layers.sort_unstable();
    layers.dedup();
    if layers.is_empty() {
        return Err(crate::Error::InvalidConfig(
            "freeze layer list requires at least one layer".to_string(),
        ));
    }
    Ok(layers)
}
