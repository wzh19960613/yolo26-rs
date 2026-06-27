use super::*;

impl Model {
    /// Saves current trainable variables to safetensors.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        Ok(self.varmap.save(path)?)
    }

    /// Saves current trainable variables to an official `.pt` checkpoint that
    /// `torch.load` can read (requires the `pt` feature).
    ///
    /// When class names were supplied through [`Self::new_with_class_names`] or
    /// [`Self::set_class_names`], they are embedded as official `model.names`.
    /// For custom class counts without names, generated `class_{id}` labels are
    /// written so `model.nc` and `model.names` stay consistent.
    #[cfg(feature = "pt")]
    pub fn save_pt(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        let tensors = self.tensor_map()?;
        let metadata = self.class_metadata();
        crate::pt_loader::save_pt_with_class_metadata(
            path,
            &tensors,
            self.task_str(),
            self.scale(),
            Some(&metadata),
        )
    }

    /// Saves current trainable variables to an official `.pt` checkpoint and
    /// embeds the provided class names.
    ///
    /// `names.len()` must match the model's `labels_count`.
    #[cfg(feature = "pt")]
    pub fn save_pt_with_names(
        &self,
        path: impl AsRef<Path>,
        names: &[String],
    ) -> crate::Result<()> {
        super::validate_class_names(self.labels_count, Some(names))?;
        let tensors = self.tensor_map()?;
        crate::pt_loader::save_pt_with_names(path, &tensors, self.task_str(), self.scale(), names)
    }

    /// Saves current trainable variables to an official `.pt` checkpoint using
    /// an existing `.pt` file as the full storage template. Non-trainable
    /// buffers absent from the native model are preserved from the template.
    #[cfg(feature = "pt")]
    pub fn save_pt_with_template_file(
        &self,
        path: impl AsRef<Path>,
        template_path: impl AsRef<Path>,
    ) -> crate::Result<()> {
        let tensors = self.tensor_map()?;
        crate::pt_loader::save_pt_with_template_file(path, &tensors, template_path)
    }

    pub(crate) fn tensor_map(&self) -> crate::Result<HashMap<String, Tensor>> {
        Ok(self
            .named_variables()?
            .into_iter()
            .map(|(name, var)| (name, var.as_tensor().clone()))
            .collect())
    }

    #[cfg(feature = "pt")]
    pub(crate) fn class_metadata(&self) -> crate::pt_loader::PtClassMetadata<'_> {
        crate::pt_loader::PtClassMetadata {
            labels_count: self.labels_count,
            names: self.class_names.as_deref(),
        }
    }
}
