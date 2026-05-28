# Zenoh / robotics

Veld is robotics-native. The `zenoh` feature enables pub/sub transport via
[Zenoh](https://zenoh.io/) — Eclipse Foundation's protocol for distributed
edge systems, with native ROS2 bridges.

```sh
cargo build --release --features zenoh
```

## What this gives you

| Capability | Description |
|---|---|
| Distributed memory | Multiple veld nodes share memory state across a Zenoh network |
| ROS2 bridge | Memories can be published as ROS2 topics and vice versa |
| Embedded device deployment | Veld runs on ARM, RISC-V, microcontrollers (with `no_std` paths planned) |
| Offline-first | Nodes operate disconnected; sync when reconnected |

Source: [`src/zenoh_transport/`](https://github.com/Portll/veld/tree/main/src/zenoh_transport).

## Architecture

Each veld instance is a Zenoh participant. Memories can be:

- **Local-only** — never published (default for sensitive personal data).
- **Published** — broadcast to a Zenoh key-expression for other participants.
- **Subscribed** — incoming memories from peers are ingested through the
  same path as local `remember` calls.

The Zenoh embedder cache ([`src/embeddings/zenoh_embedder.rs`](https://github.com/Portll/veld/blob/main/src/embeddings/zenoh_embedder.rs))
shares embedding results across nodes — embedding "auth middleware uses
API keys" once on one node makes it available to all without re-computation.

## Use cases

- **Robot fleets** — many robots, shared experience pool.
- **Edge analytics** — sensor data captured locally, semantic memory shared
  with a base station.
- **Federated personal memory** — your phone, laptop, and home server all
  contribute to one memory; sync over Zenoh; works offline.

## Set the shared session

For nodes that want to reuse Zenoh sessions across veld subsystems:

```rust
veld::memory::set_shared_zenoh_session(session);
```

This is set once at startup; the embedder cache and transport layer share
the same session to avoid duplicate connections.

## See also

- [Deploying](deploying.md) — embedded deployment notes
- Veld's robotics-track work lives partially under `tui/` (dashboard)
