# Licensing

veld is published for evaluation. The licence in [LICENSE](LICENSE) is the only grant
that comes with the repository; it lets you install, run, read and assess veld on your
own hardware, and nothing more.

## 1. Evaluation licence (the default)

[LICENSE](LICENSE) is the Veld Evaluation Licence, version 1.0. If you do nothing else,
this is the licence you hold. It is personal, revocable and non-transferable. It does
not permit commercial use, modification, derivative works, redistribution, hosting, or
use of veld to provide a service to anyone else.

## 2. PolyForm Internal Use License 1.0.0, by agreement

Internal business use is licensed under the
[PolyForm Internal Use License 1.0.0](https://polyformproject.org/licenses/internal-use/1.0.0/),
a fixed, published text, granted by a one-page written agreement naming the licensee,
the components, the versions, the term and the fee. There is no negotiation of the
licence text.

PolyForm Internal Use does not cover hosting for third parties or redistribution. Those
remain a matter for a separate agreement.

Contact: **john@portll.net**

## 3. Stated intention: AGPL-3.0-or-later at veld 2.0

Portll intends to release veld version 2.0 under the GNU Affero General Public License,
version 3 or later. This is a statement of present intention, recorded here so that the
direction is public. It is not a licence, an offer or a promise; it grants no right in
any version, and it may change. Until a release is made under that licence, the
evaluation licence governs.

## Upstream components

veld is a derivative work of [shodh-memory](https://github.com/varun29ankuS/shodh-memory)
by Varun Sharma and contributors, licensed under the **Apache License 2.0**.

Those upstream portions remain under Apache-2.0 and remain available from the upstream
repository under that licence. Apache-2.0 section 2 grants the right to sublicense
derivative works, and section 4 permits different licence terms for the derivative work
as a whole, provided its conditions are met for the Apache-licensed material; that is
the basis for offering the combined work under the terms above. The evaluation licence
names those portions as an Excluded Component and does not restrict the Apache-2.0
grant in them. See [NOTICE](NOTICE) for the full attribution, which must be retained in
redistributions under Apache-2.0 section 4.

## Third-party dependencies

All 719 crates in `Cargo.lock` were resolved and checked. Every one is permissively
licensed (MIT, Apache-2.0, ISC, BSD, Zlib, Unicode-3.0, CC0 or CDLA-Permissive-2.0),
with these dual/triple-licensed cases, where the permissive option is elected:

| Dependency | Declared | Elected |
|---|---|---|
| zenoh (27 crates), stabby (3 crates) | `EPL-2.0 OR Apache-2.0` | Apache-2.0 |
| `option-ext`, `webpki-roots` | `MPL-2.0` | MPL-2.0 (unmodified) |
| `ryu`, `wasite`, `whoami` | `Apache-2.0 OR BSL-1.0 [OR MIT]` | Apache-2.0 |
| `htmlescape` | `Apache-2.0 / MIT / MPL-2.0` | MIT |
| `r-efi` | `MIT OR Apache-2.0 OR LGPL-2.1-or-later` | MIT |

No dependency is GPL, AGPL, SSPL, Elastic, PolyForm, proprietary or undeclared.

Two notes on things that read as risk and are not:

- **`BSL-1.0` is the Boost Software License**, not the Business Source License. Boost is
  permissive. The two abbreviations differ by one letter and mean opposite things.
- **MPL-2.0** is file-level weak copyleft. The obligation applies only to modifications
  of those files, which this project does not make.

## History

While this repository was private, its tree carried the Business Source License 1.1 and
then the AGPL-3.0-or-later text. No public release was made under either. Any copy
obtained under those terms keeps them; relicensing is not retroactive and does not
revoke a grant already made. The repository was first made public under the evaluation
licence on 2026-10-11.
