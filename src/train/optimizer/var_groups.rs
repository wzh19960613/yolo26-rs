use candle_core::Var;

/// Optimizer parameter group role aligned with Ultralytics grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OptimizerGroupRole {
    /// Weight parameters with configured decay.
    Main,
    /// Normalization or logit-scale parameters with no decay.
    NoDecay,
    /// Bias parameters with no decay and bias warmup.
    Bias,
}

pub(crate) struct OptimizerVarGroup {
    pub(crate) role: OptimizerGroupRole,
    pub(crate) lr_scale: f64,
    pub(crate) vars: Vec<(String, Var)>,
}

pub(crate) fn optimizer_var_groups(
    named_vars: Vec<(String, Var)>,
    use_musgd_high_lr: bool,
) -> Vec<OptimizerVarGroup> {
    let mut groups = Vec::<OptimizerVarGroup>::new();
    for (name, var) in named_vars {
        let rank = var.dims().len();
        let role = variable_role(&name, rank);
        let lr_scale = if use_musgd_high_lr && is_musgd_high_lr_variable(&name) {
            3.0
        } else {
            1.0
        };
        push_group_var(&mut groups, role, lr_scale, name, var);
    }
    groups
}

fn push_group_var(
    groups: &mut Vec<OptimizerVarGroup>,
    role: OptimizerGroupRole,
    lr_scale: f64,
    name: String,
    var: Var,
) {
    if let Some(group) = groups
        .iter_mut()
        .find(|group| group.role == role && group.lr_scale == lr_scale)
    {
        group.vars.push((name, var));
    } else {
        groups.push(OptimizerVarGroup {
            role,
            lr_scale,
            vars: vec![(name, var)],
        });
    }
}

fn variable_role(name: &str, rank: usize) -> OptimizerGroupRole {
    if name.contains("bias") {
        OptimizerGroupRole::Bias
    } else if is_no_decay_variable(name) || rank < 2 {
        // Matches the official rule: parameters with fewer than 2 dimensions
        // (biases, normalization scales, 1D projections) get no weight decay,
        // alongside the name-based norm/bn/logit_scale heuristic.
        OptimizerGroupRole::NoDecay
    } else {
        OptimizerGroupRole::Main
    }
}

fn is_no_decay_variable(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("logit_scale")
        || lower.contains("norm")
        || lower.contains(".bn")
        || lower.contains("_bn")
        || lower.starts_with("bn.")
}

fn is_musgd_high_lr_variable(name: &str) -> bool {
    (name.contains("23") && name.contains("cv3"))
        || name.contains("proto.semseg")
        || name.contains("SemanticSegment")
}
