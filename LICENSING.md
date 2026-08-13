# Licensing

veld is dual-licensed. You may use it under **either** of the following, at your option.

## 1. GNU Affero General Public License v3.0 or later

The full text is in [LICENSE](LICENSE). This is the default: if you do nothing, this is
the licence you are using veld under.

AGPL-3.0 is an OSI-approved open source licence. You may use, modify and distribute veld
freely under its terms. The obligation that distinguishes AGPL from GPL is **section 13**:
if you modify veld and let users interact with it over a network, you must offer those
users the corresponding source of your modified version.

Running an unmodified veld — locally, on your own hardware, inside your own product's
network — triggers no source obligation. Modifying it and exposing the modified version
as a network service does.

## 2. Commercial licence

If the AGPL's terms do not suit your deployment, a commercial licence is available from
Portll. It grants the same software free of the section 13 network-source obligation and
free of the copyleft requirement on your own code.

You will typically want the commercial licence if you intend to:

- embed veld in a proprietary product you distribute without offering source;
- offer a modified veld as a hosted or network-accessible service without publishing
  your modifications;
- link veld into a codebase you cannot or will not release under AGPL-3.0.

Contact: **john@portll.net**

Nothing in the commercial licence removes anyone else's rights under the AGPL, and
choosing it does not make the AGPL-licensed distribution unavailable to others.

## Upstream components

veld is a derivative work of [shodh-memory](https://github.com/varun29ankuS/shodh-memory)
by Varun Sharma and contributors, licensed under the **Apache License 2.0**.

Those upstream portions remain under Apache-2.0 and remain available from the upstream
repository under that licence. Apache-2.0 section 2 grants the right to sublicense
derivative works, which is the basis for offering the combined work under the terms
above; Apache-2.0 is also one-way compatible with AGPL-3.0, so those portions may be
conveyed as part of an AGPL work. See [NOTICE](NOTICE) for the full attribution, which
must be retained in redistributions under Apache-2.0 section 4.

## Third-party dependencies

All 719 crates in `Cargo.lock` were resolved and checked. Every one is permissively
licensed — MIT, Apache-2.0, ISC, BSD, Zlib, Unicode-3.0, CC0 or CDLA-Permissive-2.0 —
with these dual/triple-licensed cases, where the permissive option is elected:

| Dependency | Declared | Elected |
|---|---|---|
| zenoh (27 crates), stabby (3 crates) | `EPL-2.0 OR Apache-2.0` | Apache-2.0 |
| `option-ext`, `webpki-roots` | `MPL-2.0` | MPL-2.0 (AGPL-compatible; unmodified) |
| `ryu`, `wasite`, `whoami` | `Apache-2.0 OR BSL-1.0 [OR MIT]` | Apache-2.0 |
| `htmlescape` | `Apache-2.0 / MIT / MPL-2.0` | MIT |
| `r-efi` | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` | MIT |

No dependency is GPL, AGPL, SSPL, Elastic, PolyForm, proprietary or undeclared.

Two notes on things that read as risk and are not:

- **`BSL-1.0` is the Boost Software License**, not the Business Source License. Boost is
  permissive and GPL/AGPL-compatible. The two abbreviations differ by one letter and mean
  opposite things.
- **MPL-2.0** is file-level weak copyleft and is explicitly GPL/AGPL-compatible via its
  section 3.3. The obligation applies only to modifications of those files, which this
  project does not make.

## History

veld was previously distributed under the Business Source License 1.1 (BUSL-1.1), a
source-available licence that is not OSI-approved. Versions released under BUSL-1.1
remain available to their recipients under those terms — relicensing is not retroactive
and does not revoke any grant already made. Releases from this change forward are
AGPL-3.0-or-later or commercial.
