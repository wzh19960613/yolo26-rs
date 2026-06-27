use super::*;

impl Model {
    /// Creates a trainable model with freshly initialized variables.
    pub fn new(config: ModelConfig) -> crate::Result<Self> {
        Self::new_inner(config, None)
    }

    /// Creates a trainable model with class names embedded in later `.pt`
    /// saves.
    ///
    /// `names.len()` must match the config's `labels_count`; the names are
    /// written as official `model.names` metadata by [`Self::save_pt`].
    pub fn new_with_class_names(config: ModelConfig, names: Vec<String>) -> crate::Result<Self> {
        Self::new_inner(config, Some(names))
    }

    fn new_inner(config: ModelConfig, class_names: Option<Vec<String>>) -> crate::Result<Self> {
        config.validate()?;
        validate_class_names(config.labels_count(), class_names.as_deref())?;
        let varmap = VarMap::new();
        let dtype = config.dtype();
        let device = config.device();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device).pp("model");
        let network = match &config {
            ModelConfig::Detect(config) => {
                TrainableNetwork::Detect(Box::new(crate::detect::network::load(vb, config)?))
            }
            ModelConfig::Classify(config) => TrainableNetwork::Classify(Box::new(
                crate::classify::network::Network::load(vb, config)?,
            )),
            ModelConfig::Segment(config) => {
                TrainableNetwork::Segment(Box::new(crate::segment::network::load(vb, config)?))
            }
            ModelConfig::Pose(config) => {
                TrainableNetwork::Pose(Box::new(crate::pose::network::load(vb, config)?))
            }
            ModelConfig::Semantic(config) => TrainableNetwork::Semantic(Box::new(
                crate::semantic::network::Network::load(vb, config)?,
            )),
            ModelConfig::Obb(config) => {
                TrainableNetwork::Obb(Box::new(crate::obb::network::load(vb, config)?))
            }
        };
        Ok(Self {
            varmap,
            network,
            task: config.task(),
            dtype,
            device,
            scale: config.scale(),
            labels_count: config.labels_count(),
            class_names,
        })
    }

    /// Creates a trainable model and initializes matching variables from safetensors bytes.
    pub fn from_safetensors(weights: Vec<u8>, config: ModelConfig) -> crate::Result<Self> {
        let mut model = Self::new(config)?;
        model.load_safetensors(weights)?;
        Ok(model)
    }

    /// Creates a trainable model and initializes matching variables from a safetensors file.
    pub fn from_safetensors_file(
        path: impl AsRef<Path>,
        config: ModelConfig,
    ) -> crate::Result<Self> {
        let mut model = Self::new(config)?;
        model.load_safetensors_file(path)?;
        Ok(model)
    }

    /// Creates a trainable model and initializes matching variables from an
    /// official `.pt` checkpoint.
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<Path>, config: ModelConfig) -> crate::Result<Self> {
        let mut model = Self::new(config)?;
        model.load_pt_file(path)?;
        Ok(model)
    }

    /// Returns the task represented by this model.
    pub const fn task(&self) -> Task {
        self.task
    }

    /// Returns the lowercase task identifier used in `.pt` template names.
    pub fn task_str(&self) -> &'static str {
        self.task.as_str()
    }

    /// Returns the model scale declared when the model was built.
    pub const fn scale(&self) -> crate::Scale {
        self.scale
    }

    /// Returns all trainable variables.
    pub fn variables(&self) -> Vec<Var> {
        self.varmap.all_vars()
    }

    /// Returns named trainable variables in deterministic order.
    pub fn named_variables(&self) -> crate::Result<Vec<(String, Var)>> {
        let data = self.varmap.data().lock().map_err(|_| {
            crate::Error::InvalidConfig("failed to lock trainable variable map".to_string())
        })?;
        let mut vars = data
            .iter()
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect::<Vec<_>>();
        vars.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(vars)
    }

    /// Returns variables whose names satisfy a predicate.
    pub fn variables_with_name_filter(
        &self,
        mut filter: impl FnMut(&str) -> bool,
    ) -> crate::Result<Vec<Var>> {
        Ok(self
            .named_variables()?
            .into_iter()
            .filter_map(|(name, var)| filter(&name).then_some(var))
            .collect())
    }

    /// Returns the model dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the model device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns class names that will be embedded in `.pt` checkpoints.
    pub fn class_names(&self) -> Option<&[String]> {
        self.class_names.as_deref()
    }

    /// Sets class names to embed in later `.pt` checkpoints.
    ///
    /// `names.len()` must match the model's `labels_count`.
    pub fn set_class_names(&mut self, names: Vec<String>) -> crate::Result<()> {
        validate_class_names(self.labels_count, Some(&names))?;
        self.class_names = Some(names);
        Ok(())
    }

    /// Loads matching variables from safetensors bytes.
    pub fn load_safetensors(&mut self, weights: Vec<u8>) -> crate::Result<LoadReport> {
        let tensors = candle_core::safetensors::load_buffer(&weights, &self.device)?;
        self.load_tensor_map(tensors)
    }

    /// Loads matching variables from a safetensors file.
    pub fn load_safetensors_file(&mut self, path: impl AsRef<Path>) -> crate::Result<LoadReport> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        self.load_tensor_map(tensors)
    }

    /// Loads matching variables from an official `.pt` checkpoint.
    #[cfg(feature = "pt")]
    pub fn load_pt_file(&mut self, path: impl AsRef<Path>) -> crate::Result<LoadReport> {
        let tensors = crate::pt_loader::load_pt_to_tensors(path, &self.device)?;
        self.load_tensor_map(tensors)
    }

    /// Loads matching variables from a weights file, dispatching on the
    /// extension: `.pt` (official checkpoint) or `.safetensors`.
    pub fn load_weights_file(&mut self, path: impl AsRef<Path>) -> crate::Result<LoadReport> {
        let path_ref = path.as_ref();
        let is_pt = path_ref
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pt"));
        if is_pt {
            #[cfg(feature = "pt")]
            {
                self.load_pt_file(path_ref)
            }
            #[cfg(not(feature = "pt"))]
            {
                Err(crate::Error::InvalidConfig(
                    "loading .pt requires the 'pt' feature".to_string(),
                ))
            }
        } else {
            self.load_safetensors_file(path_ref)
        }
    }
}
