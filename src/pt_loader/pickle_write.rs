//! Minimal pickle protocol-2 writer for Candle's parsed PyTorch objects.

use std::io::Write;

use candle_core::pickle::Object;

use crate::Result;

/// Serializes a parsed pickle object back to protocol-2 bytes.
pub(crate) fn to_vec(object: &Object) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x80, 0x02]);
    write_object(object, &mut out, false)?;
    out.push(b'.');
    Ok(out)
}

fn write_object(object: &Object, out: &mut Vec<u8>, allow_newobj: bool) -> Result<()> {
    match object {
        Object::Class {
            module_name,
            class_name,
        } => write_global(module_name, class_name, out),
        Object::Int(value) => write_int(*value as i64, out),
        Object::Long(value) => write_int(*value, out),
        Object::Float(value) => {
            out.push(b'G');
            out.extend_from_slice(&value.to_be_bytes());
            Ok(())
        }
        Object::Unicode(value) => write_unicode(value, out),
        Object::Bool(value) => {
            out.push(if *value { 0x88 } else { 0x89 });
            Ok(())
        }
        Object::None => {
            out.push(b'N');
            Ok(())
        }
        Object::Tuple(items) => write_tuple(items, out),
        Object::List(items) => write_list(items, out),
        Object::Dict(pairs) => write_dict(pairs, out),
        Object::Reduce { callable, args } => {
            write_object(callable, out, false)?;
            write_object(args, out, false)?;
            out.push(if allow_newobj && reduce_as_newobj(callable) {
                0x81
            } else {
                b'R'
            });
            Ok(())
        }
        Object::Build { callable, args } => {
            write_object(callable, out, true)?;
            write_object(args, out, false)?;
            out.push(b'b');
            Ok(())
        }
        Object::PersistentLoad(inner) => {
            write_object(inner, out, false)?;
            out.push(b'Q');
            Ok(())
        }
        Object::Mark => Err(crate::Error::InvalidConfig(
            "cannot serialize pickle marker object".to_string(),
        )),
    }
}

fn write_global(module_name: &str, class_name: &str, out: &mut Vec<u8>) -> Result<()> {
    out.push(b'c');
    out.write_all(module_name.as_bytes())?;
    out.push(b'\n');
    out.write_all(class_name.as_bytes())?;
    out.push(b'\n');
    Ok(())
}

fn reduce_as_newobj(callable: &Object) -> bool {
    let Object::Class {
        module_name,
        class_name,
    } = callable
    else {
        return false;
    };
    !matches!(
        (module_name.as_str(), class_name.as_str()),
        ("torch._utils", "_rebuild_tensor_v2")
            | ("torch._utils", "_rebuild_parameter")
            | ("torch._tensor", "_rebuild_from_type_v2")
    )
}

fn write_unicode(value: &str, out: &mut Vec<u8>) -> Result<()> {
    let bytes = value.as_bytes();
    let len = u32::try_from(bytes.len()).map_err(|_| {
        crate::Error::InvalidConfig("pickle unicode string is too large".to_string())
    })?;
    out.push(b'X');
    out.extend_from_slice(&len.to_le_bytes());
    out.write_all(bytes)?;
    Ok(())
}

fn write_int(value: i64, out: &mut Vec<u8>) -> Result<()> {
    if (0..=u8::MAX as i64).contains(&value) {
        out.push(b'K');
        out.push(value as u8);
    } else if (0..=u16::MAX as i64).contains(&value) {
        out.push(b'M');
        out.extend_from_slice(&(value as u16).to_le_bytes());
    } else if (i32::MIN as i64..=i32::MAX as i64).contains(&value) {
        out.push(b'J');
        out.extend_from_slice(&(value as i32).to_le_bytes());
    } else {
        let bytes = long1_bytes(value);
        out.push(0x8a);
        out.push(u8::try_from(bytes.len()).map_err(|_| {
            crate::Error::InvalidConfig("pickle long integer is too large".to_string())
        })?);
        out.write_all(&bytes)?;
    }
    Ok(())
}

fn long1_bytes(value: i64) -> Vec<u8> {
    let mut bytes = value.to_le_bytes().to_vec();
    while bytes.len() > 1 {
        let last = *bytes.last().unwrap();
        let prev = bytes[bytes.len() - 2];
        let redundant_positive = last == 0x00 && prev & 0x80 == 0;
        let redundant_negative = last == 0xff && prev & 0x80 != 0;
        if redundant_positive || redundant_negative {
            bytes.pop();
        } else {
            break;
        }
    }
    bytes
}

fn write_tuple(items: &[Object], out: &mut Vec<u8>) -> Result<()> {
    match items {
        [] => out.push(b')'),
        [one] => {
            write_object(one, out, false)?;
            out.push(0x85);
        }
        [one, two] => {
            write_object(one, out, false)?;
            write_object(two, out, false)?;
            out.push(0x86);
        }
        [one, two, three] => {
            write_object(one, out, false)?;
            write_object(two, out, false)?;
            write_object(three, out, false)?;
            out.push(0x87);
        }
        _ => {
            out.push(b'(');
            for item in items {
                write_object(item, out, false)?;
            }
            out.push(b't');
        }
    }
    Ok(())
}

fn write_list(items: &[Object], out: &mut Vec<u8>) -> Result<()> {
    out.push(b']');
    if !items.is_empty() {
        out.push(b'(');
        for item in items {
            write_object(item, out, false)?;
        }
        out.push(b'e');
    }
    Ok(())
}

fn write_dict(pairs: &[(Object, Object)], out: &mut Vec<u8>) -> Result<()> {
    out.push(b'}');
    if !pairs.is_empty() {
        out.push(b'(');
        // Candle's pickle stack reports SETITEMS dictionaries in stack-pop
        // order. Reversing here preserves the original Python insertion order,
        // which is required for nn.Module._modules / ModuleList execution.
        for (key, value) in pairs.iter().rev() {
            write_object(key, out, false)?;
            write_object(value, out, false)?;
        }
        out.push(b'u');
    }
    Ok(())
}
