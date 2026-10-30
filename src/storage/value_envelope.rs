//! Compressed envelope for stored memory records, shared by both engines.
//!
//! Both engines read it; neither writes it yet. The writer lands once every server that may
//! open a store can read the envelope, so a rollback never meets a record it cannot decode.
//!
//! Layout: `VEZ1` | raw length (u32 LE) | zstd frame (level 3) of the record's bincode.
//! A record is wrapped only when the envelope is smaller than the bincode it carries, so an
//! incompressible record stays as it was. Wrapping happens before record encryption (a
//! ciphertext does not compress) and unwrapping after decryption.
//!
//! No record written before the envelope can be mistaken for one: every bincode layout of
//! `Memory`, current or legacy, starts with the length prefix of the 16-byte id (`0x10`), a
//! versioned record with `SHO` and a sealed one with `ENC\0`.

use std::borrow::Cow;

use anyhow::{anyhow, Context, Result};

const MAGIC: &[u8; 4] = b"VEZ1";
const HEADER_LEN: usize = MAGIC.len() + 4;
const ZSTD_LEVEL: i32 = 3;
/// The decode bound for a stored record, the same as bincode's allocation limit on reads.
const MAX_RAW_LEN: usize = 64 * 1024 * 1024;

/// `plain`, enveloped when that makes it smaller.
pub fn wrap(plain: Vec<u8>) -> Vec<u8> {
    let Ok(raw_len) = u32::try_from(plain.len()) else {
        return plain;
    };
    let Ok(body) = zstd::bulk::compress(&plain, ZSTD_LEVEL) else {
        return plain;
    };
    if HEADER_LEN + body.len() >= plain.len() {
        return plain;
    }
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&raw_len.to_le_bytes());
    out.extend_from_slice(&body);
    out
}

/// Whether `data` carries the envelope header.
pub fn is_enveloped(data: &[u8]) -> bool {
    data.starts_with(MAGIC)
}

/// The bytes inside `data`'s envelope, or `data` itself when it has none. An envelope that
/// does not decode to exactly its declared length is an error, never a plain record.
pub fn unwrap(data: &[u8]) -> Result<Cow<'_, [u8]>> {
    if !is_enveloped(data) {
        return Ok(Cow::Borrowed(data));
    }
    let header = data
        .get(MAGIC.len()..HEADER_LEN)
        .ok_or_else(|| anyhow!("record envelope header is truncated"))?;
    let raw_len = u32::from_le_bytes(header.try_into().expect("four header bytes")) as usize;
    if raw_len > MAX_RAW_LEN {
        return Err(anyhow!(
            "record envelope declares {raw_len} bytes (limit {MAX_RAW_LEN})"
        ));
    }
    let plain = zstd::bulk::decompress(&data[HEADER_LEN..], raw_len)
        .context("record envelope does not decompress")?;
    if plain.len() != raw_len {
        return Err(anyhow!(
            "record envelope holds {} bytes, declares {raw_len}",
            plain.len()
        ));
    }
    Ok(Cow::Owned(plain))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repetitive(len: usize) -> Vec<u8> {
        b"the harbour crane is serviced in spring; "
            .iter()
            .copied()
            .cycle()
            .take(len)
            .collect()
    }

    #[test]
    fn a_compressible_record_round_trips_smaller() {
        let plain = repetitive(4096);
        let wrapped = wrap(plain.clone());
        assert!(is_enveloped(&wrapped));
        assert!(wrapped.len() < plain.len() / 4, "{} bytes", wrapped.len());
        assert_eq!(unwrap(&wrapped).unwrap().as_ref(), plain.as_slice());
    }

    #[test]
    fn a_record_that_would_not_shrink_is_kept_as_is() {
        let tiny = vec![0x10, 1, 2, 3];
        assert_eq!(wrap(tiny.clone()), tiny);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
        let noise: Vec<u8> = (0..2048).map(|_| rand::Rng::gen::<u8>(&mut rng)).collect();
        assert_eq!(wrap(noise.clone()), noise);
    }

    #[test]
    fn a_record_without_the_header_passes_through() {
        let plain = vec![0x10; 64];
        assert!(matches!(unwrap(&plain).unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn a_damaged_envelope_is_an_error() {
        let mut wrapped = wrap(repetitive(4096));
        let last = wrapped.len() - 1;
        wrapped[last] ^= 0xff;
        assert!(unwrap(&wrapped).is_err());
        assert!(unwrap(&wrapped[..6]).is_err());
        let mut lying = wrap(repetitive(4096));
        lying[4..8].copy_from_slice(&4000u32.to_le_bytes());
        assert!(unwrap(&lying).is_err());
        let mut huge = wrap(repetitive(4096));
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(unwrap(&huge).is_err());
    }
}
