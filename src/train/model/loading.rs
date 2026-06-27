use super::*;

impl Model {
    pub(crate) fn load_tensor_map(
        &mut self,
        tensors: HashMap<String, Tensor>,
    ) -> crate::Result<LoadReport> {
        let mut loaded = 0;
        let mut missing_names = Vec::new();
        let mut skipped_names = Vec::new();
        let data = self.varmap.data().lock().map_err(|_| {
            crate::Error::InvalidConfig("failed to lock trainable variable map".to_string())
        })?;
        let mut entries = data.iter().collect::<Vec<_>>();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (name, var) in entries {
            match tensors.get(name) {
                Some(tensor) if tensor.shape() == var.shape() => {
                    var.set(&tensor.to_dtype(var.dtype())?)?;
                    loaded += 1;
                }
                Some(_) => skipped_names.push(name.clone()),
                None => missing_names.push(name.clone()),
            }
        }
        align_skipped_one_to_one_class_heads(&data, &skipped_names)?;
        Ok(LoadReport {
            loaded,
            missing: missing_names.len(),
            skipped: skipped_names.len(),
            missing_names,
            skipped_names,
        })
    }
}

fn align_skipped_one_to_one_class_heads(
    data: &HashMap<String, Var>,
    skipped_names: &[String],
) -> crate::Result<()> {
    let skipped = skipped_names
        .iter()
        .collect::<std::collections::HashSet<_>>();
    for name in skipped_names {
        if !name.contains(".one2one_cv3.") {
            continue;
        }
        let source = name.replace(".one2one_cv3.", ".cv3.");
        if !skipped.contains(&source) {
            continue;
        }
        let Some(source_var) = data.get(&source) else {
            continue;
        };
        let Some(target_var) = data.get(name) else {
            continue;
        };
        if source_var.shape() == target_var.shape() {
            target_var.set(source_var.as_tensor())?;
        }
    }
    Ok(())
}
