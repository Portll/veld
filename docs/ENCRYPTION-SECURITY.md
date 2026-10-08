# Encryption-at-rest — confidentiality boundary (v2)

This states **exactly** what the v2 encryption protects and what it does not, so a
deployment can make an informed sign-off. Confidentiality is meaningful only with
this boundary understood.

## Encrypted (confidential at rest)
- **The full `Memory` record** in the primary CF — content, summary, tags, entity
  refs, metadata, **and the record's embedding vector** — via record-level
  XChaCha20-Poly1305 (envelope KEK→DEK; see `src/keystore.rs`).
- **Exact-match index terms** (tag, entity, external, parent, content_hash,
  episode, robot, mission, action) are **blinded** (HMAC) in `memory_index` — the
  plaintext term does not appear on disk (equality/frequency still leak).
- The keystore: master key wrapped per provider; integrity-MAC'd; rollback-guarded.

## NOT encrypted (known boundary — by necessity or by scope)
1. **The ANN vector index (similarity-search copy of embeddings).** Approximate
   nearest-neighbour search requires operating on the vectors in the clear — you
   cannot run cosine similarity over ciphertext. This is a *fundamental* limit, the
   same class as private nearest-neighbour search (research-grade; see PIR §9).
   **Mitigation:** the *authoritative* embedding lives encrypted inside the Memory
   record; the ANN index is a derived, rebuildable, plaintext search structure.
   Deployments needing vector confidentiality must run in a trusted execution
   environment or adopt private-ANN — out of scope here.
2. **Vector-mapping entries** (`memory_id ↔ vector_id`) — low-sensitivity
   identifiers; currently plaintext.
3. **Range/ordered index fields** (date / type / importance / geo / reward, and
   the `seq` ordering component of `episode_seq`) — plaintext; blinding them
   breaks range/order/proximity queries (design B-3). Note: the *episode_id* in
   `episode_seq` IS blinded (it is an exact-match term, like the `episode` index);
   only the sequence number is exposed.
4. **Sibling column families** — feedback, files, prospective, todos. These hold
   user content and are **currently plaintext**. Encrypting them is a tractable but
   separate integration (route each store's blob through the record cryptor); it is
   **deferred** and must be either completed or explicitly accepted before a
   deployment that stores sensitive data in those subsystems.

## Memory-resident plaintext
Decrypted records, the active DEK, and the master key live in process memory while
the service runs (necessary). Key-material *buffers* are zeroized after use; this is
at-rest protection, not protection against a live-process memory dump.

## Sign-off checklist before production
- [ ] Keystore file backed up (loss = unrecoverable); keyring created and its
      recovery kit stored offline (see [key-management.md](key-management.md)).
- [ ] Unseal source chosen (passphrase / env / KMS) appropriate to the threat model.
- [ ] Accept items 1–3 above as residual leakage, OR deploy compensating controls.
- [ ] Decide on item 4 (sibling CFs): complete the integration, or sign off that
      those subsystems won't hold data requiring at-rest confidentiality.
