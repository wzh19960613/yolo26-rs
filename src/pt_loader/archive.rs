//! Zip archive helpers for PyTorch `.pt` writing.

use std::io::Write;
use std::path::Path;

use crate::Result;

pub(crate) fn read_data_pkl(path: &Path) -> Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let pkl_name = zip
        .file_names()
        .map(|name| name.to_string())
        .find(|name| name.ends_with("data.pkl"))
        .ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "{} is not a PyTorch .pt checkpoint (no data.pkl entry)",
                path.display()
            ))
        })?;
    let mut reader = std::io::BufReader::new(zip.by_name(&pkl_name)?);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes)?;
    Ok(bytes)
}

pub(crate) fn write_pt_zip(
    dest: &Path,
    pkl_bytes: &[u8],
    blobs: &[(String, Vec<u8>)],
    dir_name: &str,
) -> Result<()> {
    let archive_root = dir_name.strip_suffix("/data").unwrap_or(dir_name);
    let file = std::fs::File::create(dest)?;
    let mut writer = zip::ZipWriter::new(std::io::BufWriter::new(file));
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    write_entry(
        &mut writer,
        &format!("{archive_root}/data.pkl"),
        pkl_bytes,
        stored,
    )?;
    for (path, bytes) in blobs {
        write_entry(&mut writer, path, bytes, stored)?;
    }
    write_archive_metadata(&mut writer, archive_root, stored)?;
    writer.finish()?;
    Ok(())
}

fn write_archive_metadata(
    writer: &mut zip::ZipWriter<std::io::BufWriter<std::fs::File>>,
    archive_root: &str,
    options: zip::write::SimpleFileOptions,
) -> Result<()> {
    write_entry(writer, &format!("{archive_root}/version"), b"3\n", options)?;
    write_entry(
        writer,
        &format!("{archive_root}/.format_version"),
        b"1",
        options,
    )?;
    write_entry(
        writer,
        &format!("{archive_root}/.storage_alignment"),
        b"64",
        options,
    )?;
    write_entry(
        writer,
        &format!("{archive_root}/byteorder"),
        b"little",
        options,
    )?;
    Ok(())
}

fn write_entry(
    writer: &mut zip::ZipWriter<std::io::BufWriter<std::fs::File>>,
    name: &str,
    bytes: &[u8],
    options: zip::write::SimpleFileOptions,
) -> Result<()> {
    writer.start_file(name, options)?;
    writer.write_all(bytes)?;
    Ok(())
}
