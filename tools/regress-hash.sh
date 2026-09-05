#!/usr/bin/env bash
# Engine-level regression: translate the bench corpora with both GEMM paths of
# the same smoke binary and require (1) SMMLA == ruy bytes, (2) the canonical
# hash for this platform. Exact-integer GEMM means any kernel bug changes the
# hash -- this is the correctness gate for the whole engine, no COMET needed.
#   regress-hash.sh <smoke> <enzh-config.yml> <eng.txt> [<jaen-config.yml> <jpn.txt>]
# Env: EXPECT_ENZH / EXPECT_PIVOT override the built-in table; PLATFORM=host|device
# Every canonical hash below assumes patch 0012 (requests ordered by id, not
# heap address): without it batch composition follows allocator addresses --
# ja->zh differs run to run on the host, and on Android even en->zh lands on a
# different (stable) output. Engines without 0012 cannot use this table.
set -euo pipefail
smoke=${1:?smoke}; enzh=${2:?enzh config}; eng=${3:?eng.txt}
jaen=${4:-}; jpn=${5:-}
platform=${PLATFORM:-host}

# Canonical blocking hashes (FNV-1a over the corpus output) by platform, config
# and corpus size: host runs the mbw1024 CI config, devices run the mbw512
# default. The hash covers the whole corpus and batch composition changes the
# output of individual sentences, so each corpus size has its own table entry
# (bench set grew from FLORES lines 1-150 to 1-200 on 2026-09-04).
# Host table re-baselined 2026-09-05 for patch 0018: the small-GEMM kernel
# reproduces ruy's accumulation order, which differs from Accelerate's, so the
# Accelerate host build moves (host/150 pivot happens not to); every device runs
# ruy and is unchanged. Pre-0018 host values: host/150 1728c7c863926c5c/
# 58b6667dd43d6364, host/200 3b458f7f7fe6fd68/00602a479d7c106b.
n=$(wc -l < "$eng" | tr -d ' ')
case "$platform/$n" in
  host/150)   can_enzh=0a46cfb6e339ff29; can_pivot=58b6667dd43d6364 ;;
  host/200)   can_enzh=cec5ff3b1fc29f8e; can_pivot=e9d84f82b99250ee ;;
  device/150) can_enzh=1742b57a069b1da7; can_pivot=28028fc1ed7d0099 ;;
  device/200) can_enzh=f8a315e6571cc957; can_pivot=fb3dda796b1186af ;;   # Mi 10 (865, ruy path) 2026-09-04; Mi 14 SMMLA == ruy 2026-09-05
  host/*|device/*) can_enzh=; can_pivot= ;;
  *) echo "unknown PLATFORM=$platform"; exit 2 ;;
esac
expect_enzh=${EXPECT_ENZH:-$can_enzh}; expect_pivot=${EXPECT_PIVOT:-$can_pivot}
if [ -z "$expect_enzh" ] || { [ -n "$jaen" ] && [ -z "$expect_pivot" ]; }; then
  echo "no canonical hash for $platform with a $n-line corpus; pass EXPECT_ENZH / EXPECT_PIVOT"; exit 2
fi

tmp=$(mktemp -d)
run() { # $1 env-prefix  $2 dump-prefix  $3.. configs
  local envp=$1 dump=$2; shift 2
  env $envp "$smoke" --bench --dump-prefix "$tmp/$dump" "$@" 2>&1 >/dev/null | sed -n 's/.*pass0_hash=//p'
}
status=0
check() { # name hash_smmla hash_ruy expected dumpA dumpB
  local name=$1 hs=$2 hr=$3 exp=$4
  if [ "$hs" != "$hr" ]; then echo "FAIL $name: SMMLA $hs != ruy $hr"; status=1; fi
  if ! cmp -s "$5" "$6"; then echo "FAIL $name: dumps differ"; status=1; fi
  if [ "$hs" != "$exp" ]; then echo "FAIL $name: hash $hs != canonical $exp"; status=1; fi
  [ $status -eq 0 ] && echo "OK   $name: $hs (SMMLA == ruy == canonical)"
}

hs=$(run "" enzh-smmla "$enzh" < "$eng")
hr=$(run "BERGAMOT_NO_I8MM=1" enzh-ruy "$enzh" < "$eng")
check enzh "$hs" "$hr" "$expect_enzh" "$tmp/enzh-smmla.pass0.txt" "$tmp/enzh-ruy.pass0.txt"

if [ -n "$jaen" ]; then
  hs=$(run "" pivot-smmla "$jaen" "$enzh" < "$jpn")
  hr=$(run "BERGAMOT_NO_I8MM=1" pivot-ruy "$jaen" "$enzh" < "$jpn")
  check pivot "$hs" "$hr" "$expect_pivot" "$tmp/pivot-smmla.pass0.txt" "$tmp/pivot-ruy.pass0.txt"
fi
rm -rf "$tmp"
exit $status
