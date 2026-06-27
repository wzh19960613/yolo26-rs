use super::*;

pub(crate) fn classification_class_names_in_split(split_dir: &Path) -> crate::Result<Vec<String>> {
    let mut classes = Vec::new();
    for entry in std::fs::read_dir(split_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                crate::Error::InvalidConfig(format!(
                    "classification class directory is not valid UTF-8: {}",
                    path.display()
                ))
            })?
            .to_string();
        classes.push(name);
    }
    classes.sort();
    if classes.is_empty() {
        return Err(crate::Error::InvalidConfig(format!(
            "classification split must contain class subdirectories: {}",
            split_dir.display()
        )));
    }
    Ok(classes)
}

pub(crate) fn collect_images_in_dir_inner(dir: &Path, out: &mut Vec<PathBuf>) -> crate::Result<()> {
    for entry in sorted_dir_entries(dir)? {
        let path = entry.path();
        if path.is_dir() {
            collect_images_in_dir_inner(&path, out)?;
        } else if is_supported_image(&path) {
            out.push(path);
        }
    }
    Ok(())
}

pub(crate) fn sorted_dir_entries(dir: &Path) -> crate::Result<Vec<std::fs::DirEntry>> {
    let mut entries = std::fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.path());
    Ok(entries)
}

pub(crate) fn collect_images_from_list(
    root: &Path,
    list_path: &Path,
) -> crate::Result<Vec<PathBuf>> {
    let list_dir = list_path.parent().unwrap_or(root);
    let mut out = Vec::new();
    for raw in std::fs::read_to_string(list_path)?.lines() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let path = Path::new(line);
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            let root_candidate = root.join(path);
            if root_candidate.exists() {
                root_candidate
            } else {
                list_dir.join(path)
            }
        };
        if is_supported_image(&resolved) {
            out.push(resolved);
        }
    }
    Ok(out)
}

pub(crate) fn path_has_glob(path: &Path) -> bool {
    path.components()
        .any(|component| glob_component_has_wildcard(&component.as_os_str().to_string_lossy()))
}

pub(crate) fn glob_component_has_wildcard(component: &str) -> bool {
    component.contains('*') || component.contains('?')
}

pub(crate) fn glob_component_matches(pattern: &str, text: &str) -> bool {
    let pattern = pattern.chars().collect::<Vec<_>>();
    let text = text.chars().collect::<Vec<_>>();
    let mut memo = vec![vec![None; text.len() + 1]; pattern.len() + 1];
    glob_component_matches_inner(&pattern, &text, 0, 0, &mut memo)
}

fn glob_component_matches_inner(
    pattern: &[char],
    text: &[char],
    pattern_idx: usize,
    text_idx: usize,
    memo: &mut [Vec<Option<bool>>],
) -> bool {
    if let Some(value) = memo[pattern_idx][text_idx] {
        return value;
    }
    let matched = if pattern_idx == pattern.len() {
        text_idx == text.len()
    } else if pattern[pattern_idx] == '*' {
        glob_component_matches_inner(pattern, text, pattern_idx + 1, text_idx, memo)
            || (text_idx < text.len()
                && glob_component_matches_inner(pattern, text, pattern_idx, text_idx + 1, memo))
    } else if text_idx < text.len()
        && (pattern[pattern_idx] == '?' || pattern[pattern_idx] == text[text_idx])
    {
        glob_component_matches_inner(pattern, text, pattern_idx + 1, text_idx + 1, memo)
    } else {
        false
    };
    memo[pattern_idx][text_idx] = Some(matched);
    matched
}

pub(crate) fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("txt"))
}

pub(crate) fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
}

pub(crate) fn read_rgb_image(path: &Path) -> crate::Result<crate::Image> {
    let rgb = image::open(path)
        .map_err(|err| {
            crate::Error::InvalidImage(format!("failed to decode {}: {err}", path.display()))
        })?
        .to_rgb8();
    crate::Image::new(rgb.width(), rgb.height(), rgb.into_raw())
}
