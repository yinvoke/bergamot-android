#!/usr/bin/env bash
# Confinement check for i8mm opcodes: every smmla/ummla/usmmla/sudot/usdot
# instruction in the binary must live in a symbol of the smmla namespace.
# Anything else means the raised -march leaked into a TU that runs before the
# runtime gate -> SIGILL on CPUs without i8mm.
#   check-opcodes.sh <binary-or-archive> [llvm-objdump]
set -euo pipefail
bin=${1:?binary}
objdump=${2:-$(command -v llvm-objdump || xcrun -f llvm-objdump 2>/dev/null || echo objdump)}

# Walk the disassembly: remember the current symbol, flag i8mm mnemonics.
# --mattr: without the feature the disassembler prints i8mm ops as <unknown>
# (Apple's llvm-objdump does), which would hide leaks; count those too.
"$objdump" -d --no-show-raw-insn --mattr=${MATTR:-+v9.2a} "$bin" | awk '
  /^[0-9a-f]+ <.*>:$/ { sym=$2; next }
  /^<.*>:$/ { sym=$1; next }
  /^[0-9a-fA-F]+ .*:$/ { sym=$0; next }
  $0 ~ /<unknown>/ { unknown++; if (unknown <= 3) print "UNKNOWN: " sym " :: " $0 }
  $0 ~ /[[:space:]](smmla|ummla|usmmla|sudot|usdot)([[:space:]]|$)/ {
    total++
    if (sym ~ /smmla/) ok++; else { bad++; if (bad <= 10) print "LEAK: " sym " :: " $0 }
  }
  END {
    printf("i8mm opcodes: total=%d confined=%d leaked=%d undecoded=%d\n", total, ok, bad, unknown)
    if (unknown > 0) print "WARN: " unknown " undecoded instruction(s) (non-i8mm exotics such as PAC/BTI; i8mm itself decodes under --mattr)"
    if (total == 0) { print "FAIL: no i8mm opcodes found at all (kernel not built?)"; exit 2 }
    if (bad > 0) { print "FAIL: i8mm opcodes outside the gated TU"; exit 1 }
    print "OK: all i8mm opcodes confined to smmla_gemm"
  }'
