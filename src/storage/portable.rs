//! The portable dump format — one codec, both directions.
//!
//! Export and import are the two halves of a backend cutover, and the failure
//! that matters is not either half being wrong: it is the two disagreeing. So the
//! framing lives here and both sides call it, rather than each writing its own
//! reader against a prose description of the other's writer.
//!
//! # Why there is a trailer
//!
//! A dump with no length information imports as whatever survived. Truncate the
//! last megabyte of a 2 GB export — a full disk, a killed process, a partial
//! copy — and a frame-by-frame reader hits EOF on a frame boundary and reports
//! success over a store missing everything after the cut. The migration then
//! looks complete, because "read until EOF" cannot tell a finished file from a
//! severed one.
//!
//! The trailer carries the record count and a digest of every frame byte, both
//! checked before a reader returns anything. Truncation fails on the missing
//! trailer, a short read fails on the count, and corruption fails on the digest.

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"VELDPORT";
const TRAILER: &[u8; 8] = b"VELDPEND";
const VERSION: u32 = 1;

/// One exported row. `cf` is the column family name; the default keyspace is
/// carried explicitly as `"default"` rather than as an absent value, so a reader
/// never has to guess which keyspace an unlabelled row belonged to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableRecord {
    pub cf: String,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

/// Streaming writer. Frames are hashed as they are written, so the digest costs
/// one pass and never requires the dump in memory.
pub struct PortableWriter<W: Write> {
    out: W,
    hasher: Sha256,
    count: u64,
}

impl<W: Write> PortableWriter<W> {
    pub fn new(mut out: W) -> Result<Self> {
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        Ok(Self {
            out,
            hasher: Sha256::new(),
            count: 0,
        })
    }

    pub fn write_record(&mut self, rec: &PortableRecord) -> Result<()> {
        let cf = rec.cf.as_bytes();
        let mut frame = Vec::with_capacity(12 + cf.len() + rec.key.len() + rec.value.len());
        frame.extend_from_slice(&(cf.len() as u32).to_le_bytes());
        frame.extend_from_slice(cf);
        frame.extend_from_slice(&(rec.key.len() as u32).to_le_bytes());
        frame.extend_from_slice(&rec.key);
        frame.extend_from_slice(&(rec.value.len() as u32).to_le_bytes());
        frame.extend_from_slice(&rec.value);

        self.hasher.update(&frame);
        self.out.write_all(&frame)?;
        self.count += 1;
        Ok(())
    }

    /// Writes the trailer. A dump without this is not a dump — see the module note.
    pub fn finish(mut self) -> Result<(u64, [u8; 32])> {
        let digest: [u8; 32] = self.hasher.finalize().into();
        self.out.write_all(TRAILER)?;
        self.out.write_all(&self.count.to_le_bytes())?;
        self.out.write_all(&digest)?;
        self.out.flush()?;
        Ok((self.count, digest))
    }
}

/// Reads a whole dump, refusing anything it cannot fully account for.
///
/// Deliberately not an iterator: an iterator can yield rows and only discover the
/// truncation at the end, by which point a caller has already written them
/// somewhere. Verification precedes any row reaching a consumer.
pub fn read_all<R: Read>(mut src: R) -> Result<Vec<PortableRecord>> {
    let mut buf = Vec::new();
    src.read_to_end(&mut buf).context("reading portable dump")?;

    if buf.len() < 12 {
        bail!("portable dump is {} bytes — too short to hold a header", buf.len());
    }
    if &buf[..8] != MAGIC {
        bail!("not a portable dump: magic is not VELDPORT");
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != VERSION {
        bail!("portable dump is version {version}, this build reads version {VERSION}");
    }

    // The trailer is located from the END, so a dump that was cut short simply
    // does not have one — which is the whole point.
    const TRAILER_LEN: usize = 8 + 8 + 32;
    if buf.len() < 12 + TRAILER_LEN {
        bail!("portable dump has no trailer — it is truncated, and reading to EOF would have reported success over a partial store");
    }
    let trailer_at = buf.len() - TRAILER_LEN;
    if &buf[trailer_at..trailer_at + 8] != TRAILER {
        bail!("portable dump does not end with a trailer — truncated or corrupt");
    }
    let declared_count = u64::from_le_bytes(buf[trailer_at + 8..trailer_at + 16].try_into().unwrap());
    let declared_digest: [u8; 32] = buf[trailer_at + 16..].try_into().unwrap();

    let frames = &buf[12..trailer_at];
    let actual_digest: [u8; 32] = Sha256::digest(frames).into();
    if actual_digest != declared_digest {
        bail!("portable dump digest mismatch — the frames are not the frames that were written");
    }

    let mut out = Vec::new();
    let mut i = 0usize;
    let take_u32 = |b: &[u8], i: &mut usize| -> Result<usize> {
        if *i + 4 > b.len() {
            return Err(anyhow!("frame length runs past the end of the dump"));
        }
        let n = u32::from_le_bytes(b[*i..*i + 4].try_into().unwrap()) as usize;
        *i += 4;
        Ok(n)
    };
    while i < frames.len() {
        let cf_len = take_u32(frames, &mut i)?;
        if i + cf_len > frames.len() {
            bail!("column family name runs past the end of the dump");
        }
        let cf = String::from_utf8(frames[i..i + cf_len].to_vec())
            .context("column family name is not utf-8")?;
        i += cf_len;
        let key_len = take_u32(frames, &mut i)?;
        if i + key_len > frames.len() {
            bail!("key runs past the end of the dump");
        }
        let key = frames[i..i + key_len].to_vec();
        i += key_len;
        let val_len = take_u32(frames, &mut i)?;
        if i + val_len > frames.len() {
            bail!("value runs past the end of the dump");
        }
        let value = frames[i..i + val_len].to_vec();
        i += val_len;
        out.push(PortableRecord { cf, key, value });
    }

    // Both directions. The digest proves the bytes are intact; the count proves
    // they are ALL of them, and a digest over a shorter frame region would agree
    // with itself perfectly.
    if out.len() as u64 != declared_count {
        bail!(
            "portable dump declares {declared_count} records and contains {} — refusing a partial import",
            out.len()
        );
    }
    Ok(out)
}

/// The trailer's record count and digest, without reading the frames.
///
/// For listing many dumps: `read_all` loads the whole file to answer a question the
/// last 48 bytes already hold. It still REFUSES a file with no trailer, so a
/// truncated backup is caught at list time and not only at restore time — which is
/// the point at which finding out is expensive.
pub fn read_trailer(path: &std::path::Path) -> Result<(u64, [u8; 32])> {
    use std::io::{Seek, SeekFrom};
    const TRAILER_LEN: i64 = 8 + 8 + 32;
    let mut f = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let len = f.metadata()?.len();
    if len < 12 + TRAILER_LEN as u64 {
        bail!("{} is {len} bytes — too short to hold a header and a trailer", path.display());
    }
    f.seek(SeekFrom::End(-TRAILER_LEN))?;
    let mut buf = [0u8; TRAILER_LEN as usize];
    f.read_exact(&mut buf)?;
    if &buf[..8] != TRAILER {
        bail!("{} does not end with a trailer — truncated or corrupt", path.display());
    }
    let count = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let digest: [u8; 32] = buf[16..].try_into().unwrap();
    Ok((count, digest))
}

/// A digest as the lowercase hex this codebase stores in `BackupMetadata.checksum`.
pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(cf: &str, k: &[u8], v: &[u8]) -> PortableRecord {
        PortableRecord { cf: cf.to_string(), key: k.to_vec(), value: v.to_vec() }
    }

    fn dump(records: &[PortableRecord]) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        for r in records {
            w.write_record(r).unwrap();
        }
        w.finish().unwrap();
        buf
    }

    #[test]
    fn round_trips_every_keyspace_including_empty_values() {
        let records = vec![
            rec("default", b"acl:alice:1", b"{}"),
            rec("memory_index", b"idx", b""),
            rec("kv", b"dials", b"\x00\x01\x02"),
        ];
        assert_eq!(read_all(&dump(&records)[..]).unwrap(), records);
    }

    #[test]
    fn an_empty_export_is_valid_and_reads_as_zero_records() {
        // Not the same as a truncated one, and the difference has to survive a
        // round trip: a store with nothing in it is a legitimate export.
        let bytes = dump(&[]);
        assert_eq!(read_all(&bytes[..]).unwrap().len(), 0);
    }

    #[test]
    fn the_same_records_produce_byte_identical_dumps() {
        let records = vec![rec("kv", b"a", b"1"), rec("default", b"b", b"2")];
        assert_eq!(dump(&records), dump(&records));
    }

    #[test]
    fn a_truncated_dump_is_refused_rather_than_read_to_eof() {
        // THE test this format exists for. Cutting the tail leaves whole frames
        // intact, so a frame-by-frame reader reaches EOF cleanly and reports a
        // store that is missing everything after the cut.
        let bytes = dump(&[rec("kv", b"a", b"1"), rec("kv", b"b", b"2")]);
        let cut = &bytes[..bytes.len() - 10];
        let err = read_all(cut).expect_err("a truncated dump must not read as a short store");
        assert!(
            err.to_string().contains("trailer"),
            "the error must name the missing trailer, got: {err}"
        );
    }

    #[test]
    fn a_corrupt_byte_is_refused() {
        let mut bytes = dump(&[rec("kv", b"key", b"value")]);
        let at = 20;
        bytes[at] ^= 0xff;
        let err = read_all(&bytes[..]).expect_err("a flipped byte must not import");
        assert!(err.to_string().contains("digest") || err.to_string().contains("past the end"));
    }

    #[test]
    fn a_dump_that_lies_about_its_count_is_refused() {
        // The digest alone cannot catch this: it is computed over the frames that
        // are present, so it agrees with itself no matter how few there are. The
        // count is the second witness, and it does not share that failure mode.
        let mut bytes = dump(&[rec("kv", b"a", b"1")]);
        let n = bytes.len();
        bytes[n - 40..n - 32].copy_from_slice(&99u64.to_le_bytes());
        let err = read_all(&bytes[..]).expect_err("a count that disagrees must refuse");
        assert!(err.to_string().contains("declares 99"), "got: {err}");
    }

    #[test]
    fn a_foreign_file_is_not_mistaken_for_a_dump() {
        let err = read_all(&b"this is a tarball, honestly, and it is long enough"[..])
            .expect_err("magic must be checked");
        assert!(err.to_string().contains("magic"));
    }
}

#[cfg(test)]
mod trailer_tests {
    use super::*;

    #[test]
    fn the_trailer_reader_agrees_with_a_full_read() {
        // Two ways to the same number, and they must not diverge — the cheap one is
        // what `list_backups` trusts, and a listing that disagrees with a restore is
        // worse than a slow listing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.veldport");
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        for i in 0..7u8 {
            w.write_record(&PortableRecord { cf: "kv".into(), key: vec![i], value: vec![i, i] }).unwrap();
        }
        let (count, digest) = w.finish().unwrap();
        std::fs::write(&path, &buf).unwrap();

        let (t_count, t_digest) = read_trailer(&path).unwrap();
        assert_eq!(t_count, count);
        assert_eq!(t_digest, digest);
        assert_eq!(read_all(&buf[..]).unwrap().len() as u64, t_count);
    }

    #[test]
    fn a_truncated_dump_is_refused_at_list_time_not_only_at_restore() {
        // The whole reason the cheap reader still validates: finding out a backup is
        // unusable when you come to restore it is finding out too late.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cut.veldport");
        let mut buf = Vec::new();
        let mut w = PortableWriter::new(&mut buf).unwrap();
        w.write_record(&PortableRecord { cf: "kv".into(), key: b"a".to_vec(), value: b"1".to_vec() }).unwrap();
        w.finish().unwrap();
        std::fs::write(&path, &buf[..buf.len() - 9]).unwrap();

        let err = read_trailer(&path).expect_err("a truncated dump must not list as usable");
        assert!(err.to_string().contains("trailer") || err.to_string().contains("too short"));
    }
}
