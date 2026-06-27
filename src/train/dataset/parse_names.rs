use super::*;

pub(crate) fn parse_inline_names(value: &str) -> Vec<String> {
    let value = value.trim();
    if value.starts_with('[') && value.ends_with(']') {
        return value[1..value.len() - 1]
            .split(',')
            .map(|item| unquote(item.trim()).to_string())
            .filter(|item| !item.is_empty())
            .collect();
    }
    if value.starts_with('{') && value.ends_with('}') {
        return value[1..value.len() - 1]
            .split(',')
            .filter_map(|item| {
                item.split_once(':')
                    .map(|(_, name)| unquote(name.trim()).to_string())
            })
            .filter(|item| !item.is_empty())
            .collect();
    }
    non_empty(value).into_iter().collect()
}

pub(crate) fn non_empty(value: &str) -> Option<String> {
    let value = unquote(value.trim());
    (!value.is_empty()).then(|| value.to_string())
}

pub(crate) fn parse_yaml_list_item(line: &str) -> Option<String> {
    let value = line.strip_prefix('-')?.trim();
    Some(unquote(value).to_string())
}

pub(crate) fn parse_kpt_shape(value: &str) -> crate::Result<(usize, usize)> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(crate::Error::InvalidConfig(
            "kpt_shape must use inline YAML list syntax, e.g. [17, 3]".to_string(),
        ));
    }
    let parts = value[1..value.len() - 1]
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| crate::Error::InvalidConfig(format!("invalid kpt_shape: {err}")))?;
    if parts.len() != 2 || parts[0] == 0 || !matches!(parts[1], 2 | 3) {
        return Err(crate::Error::InvalidConfig(
            "kpt_shape must be [keypoints_count, 2|3]".to_string(),
        ));
    }
    Ok((parts[0], parts[1]))
}

pub(crate) fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        .unwrap_or(value)
}

pub(crate) fn dataset_root(yaml_path: &Path, path_value: Option<&str>) -> PathBuf {
    let yaml_dir = yaml_path.parent().unwrap_or_else(|| Path::new("."));
    match path_value {
        Some(path) if Path::new(path).is_absolute() => PathBuf::from(path),
        Some(path) => yaml_dir.join(path),
        None => yaml_dir.to_path_buf(),
    }
}

pub(crate) fn resolve_dataset_path(root: &Path, value: &str) -> PathBuf {
    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

pub(crate) fn collect_image_paths(root: &Path, split_path: &Path) -> crate::Result<Vec<PathBuf>> {
    let mut images = if split_path.is_dir() {
        collect_images_in_dir(split_path)?
    } else if path_has_glob(split_path) {
        collect_images_from_glob(root, split_path)?
    } else if split_path.is_file() && is_text_file(split_path) {
        collect_images_from_list(root, split_path)?
    } else if split_path.is_file() && is_supported_image(split_path) {
        vec![split_path.to_path_buf()]
    } else {
        return Err(crate::Error::InvalidConfig(format!(
            "Ultralytics split path does not exist or is not supported: {}",
            split_path.display()
        )));
    };
    images.sort();
    Ok(images)
}

fn collect_images_from_glob(root: &Path, pattern: &Path) -> crate::Result<Vec<PathBuf>> {
    let (base, components) = split_glob_base(root, pattern);
    let mut images = Vec::new();
    collect_glob_matches(&base, &components, &mut images)?;
    Ok(images)
}

fn split_glob_base(root: &Path, pattern: &Path) -> (PathBuf, Vec<String>) {
    let mut base = PathBuf::new();
    let mut components = Vec::new();
    let mut in_glob = false;

    for component in pattern.components() {
        let label = component.as_os_str().to_string_lossy().into_owned();
        if !in_glob && !glob_component_has_wildcard(&label) {
            base.push(component.as_os_str());
        } else {
            in_glob = true;
            components.push(label);
        }
    }

    if base.as_os_str().is_empty() {
        base = root.to_path_buf();
    }

    (base, components)
}

fn collect_glob_matches(
    base: &Path,
    components: &[String],
    out: &mut Vec<PathBuf>,
) -> crate::Result<()> {
    if components.is_empty() {
        if base.is_file() && is_supported_image(base) {
            out.push(base.to_path_buf());
        }
        return Ok(());
    }

    let component = &components[0];
    let rest = &components[1..];
    if component == "**" {
        collect_glob_matches(base, rest, out)?;
        if base.is_dir() {
            for entry in sorted_dir_entries(base)? {
                let path = entry.path();
                if path.is_dir() {
                    collect_glob_matches(&path, components, out)?;
                }
            }
        }
        return Ok(());
    }

    if glob_component_has_wildcard(component) {
        if base.is_dir() {
            for entry in sorted_dir_entries(base)? {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if glob_component_matches(component, name) {
                    collect_glob_matches(&path, rest, out)?;
                }
            }
        }
        return Ok(());
    }

    collect_glob_matches(&base.join(component), rest, out)
}

pub(crate) fn collect_images_in_dir(dir: &Path) -> crate::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    collect_images_in_dir_inner(dir, &mut out)?;
    Ok(out)
}
