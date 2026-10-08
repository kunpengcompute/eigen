#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# Run on Linux AArch64; no TensorFlow wheel build or performance timing.
# Usage: bash bench/sgemm_test_suite/verify_tf220_kgemm.sh
# Optional: CXX=clang++ (otherwise test every available g++ and clang++).
# A fresh temporary directory holds all binaries and logs; it is not removed.
set -euo pipefail
if [[ $(uname -m) != aarch64 ]]; then
  echo 'ERROR: real KGEMM dispatch verification requires Linux AArch64.' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=$(mktemp -d "${TMPDIR:-/tmp}/eigen-tf220-kgemm.XXXXXXXX")
echo "Results: $out"
compilers=(g++ clang++)
if [[ -n ${CXX:-} ]]; then compilers=("$CXX"); fi
tested=0
for compiler in "${compilers[@]}"; do
  if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "SKIP unavailable compiler: $compiler"
    continue
  fi
  tested=$((tested + 1))
  prefix="$out/compiler-$tested"
  "$compiler" --version | tee "$prefix-version.txt"
  flags=(-std=c++14 -O2 -DNDEBUG -pthread -march=armv8-a+simd -DEIGEN_MAX_ALIGN_BYTES=64 -I.)
  for variant in native kgemm packed_slices raw_slices; do
    definitions=(-DEIGEN_NEON_USE_KGEMM=1)
    case "$variant" in
      native) definitions=(-DEIGEN_NEON_USE_KGEMM=0) ;;
      packed_slices) definitions+=(-DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN=32
          -DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_K=32 -DEIGEN_NEON_KGEMM_PACK_KC=17) ;;
      raw_slices) definitions+=(-DEIGEN_NEON_KGEMM_REUSE_PACKING=0
          -DEIGEN_KUNPENG_KSPLIT=1 -DEIGEN_KUNPENG_KSPLIT_THRESHOLD=32
          -DEIGEN_KUNPENG_KSPLIT_CHUNK_SIZE=17) ;;
    esac
    "$compiler" "${flags[@]}" "${definitions[@]}" \
      bench/sgemm_test_suite/kgemm_compatibility_test.cpp \
      -o "$prefix-$variant" 2>&1 | tee "$prefix-$variant-build.log"
    timeout 180 "$prefix-$variant" 2>&1 | tee "$prefix-$variant.log"
  done
  "$compiler" "${flags[@]}" -DEIGEN_USE_THREADS=1 -DEIGEN_NEON_USE_KGEMM=1 \
    -DEIGEN_BENCHMARK_KGEMM_INSTRUMENTATION=1 \
    bench/sgemm_test_suite/comprehensive_tensor_contraction_bench.cpp \
    -o "$prefix-bench" 2>&1 | tee "$prefix-bench-build.log"
  for threads in 1 2 4 8; do
    timeout 600 "$prefix-bench" --verify --kernel kgemm --layout both --mode both \
      --threads "$threads" 2>&1 | tee "$prefix-verify-t$threads.log"
  done
done
if (( tested == 0 )); then
  echo 'ERROR: no compiler found.' >&2
  exit 2
fi
echo "PASS: all available compiler configurations; logs: $out"
