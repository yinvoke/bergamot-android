# Vendor provenance

Engine sources are vendored (flattened, no submodules) from:

- **Repository**: https://github.com/mozilla/translations
- **Commit**: `df4ab487a117903b53e62cf3ac4305966fbbd2d6` (2026-08-26)
- **Subdirectory**: `inference/` (this is the actively maintained home of the
  Bergamot translator engine; the original `browsermt/bergamot-translator`
  repository is frozen — Mozilla merged `browsermt/marian-dev` into
  `inference/marian-fork` in 2025 and maintains it there for Firefox.)

Submodules of the upstream tree were materialized at their pinned commits and
copied in as plain files: `sentencepiece`, `ruy`, `simd_utils` (under
`marian-fork/src/3rd_party/`) and `ssplit-cpp` (under `3rd_party/`, trimmed).

## Deliberately NOT vendored

| Path | Reason |
|---|---|
| `src/app/` | upstream CLI; replaced by our own `tools/smoke` |
| `wasm/`, `3rd_party/emsdk` | WASM target, out of scope |
| `scripts/`, `tests/`, `docker/` | upstream dev tooling |
| `marian-fork/src/3rd_party/intgemm` | x86 int8 GEMM; this repo is ARM-only by design (ruy is the sole GEMM backend) |
| `marian-fork/src/3rd_party/{fbgemm,nccl,onnxjs,simple-websocket-server}` | x86 / CUDA / WASM / server-only; verified unnecessary for native ARM builds |
| `marian-fork/src/tests`, `regression-tests`, `examples` | upstream tests |
| `ssplit-cpp`: everything except `src/ssplit/` core (5 files), `nonbreaking_prefixes/`, in-tree `pcre2-10.39`, `cmake/`, `CMakeLists.txt`, `LICENSE.md`, `README.md` | dev tooling and its own CLI |

## Local changes

All deviations from upstream are individual commits on top of the vendor
commit, archived as `git format-patch` files in `../patches/`. Never edit
vendored files silently — one commit per logical change, then refresh
`patches/`.

## How to re-vendor (upgrade the engine)

1. Sparse-clone `mozilla/translations` at the new commit (`inference/` only);
   `git submodule update --init` for: `3rd_party/ssplit-cpp`,
   `marian-fork/src/3rd_party/{sentencepiece,ruy,simd_utils}`.
2. Overwrite `engine/` following the inclusion/exclusion table above
   (see the rsync recipe in the vendor commit message).
3. Update the commit hash at the top of this file. Commit as a single
   `vendor:` commit.
4. Replay `patches/*.patch` in order. A patch that no longer applies usually
   means upstream fixed it — verify and drop it.
5. Run the host smoke test, then the Android benchmark, before releasing.
