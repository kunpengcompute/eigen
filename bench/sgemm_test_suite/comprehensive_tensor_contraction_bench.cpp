// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
//
// Comprehensive TensorContraction benchmark with CLI options.
// Based on contraction_benchmarks_cpu.cc.
//
// Features:
//   --verify         Correctness test (compare with naive GEMM)
//   --mode ceqab     C = A * B  (benchmark default)
//   --mode cpeqab    Tensor API expression C += A * B
//   --mode both      Verify both modes (verification only)
//   --csv <file>     Read m,k,n list from CSV (first row: header, then values)
//   --layout row     RowMajor (benchmark default)
//   --layout col     ColMajor
//   --layout both    Verify both layouts (verification only)
//   --kernel neon    Validate the stock NEON GEBP backend
//   --kernel kgemm   Validate the KGemm TensorContraction kernel specialization
//   --kernel sve     Validate the 256-bit SVE2 optimized GEBP backend
//   --samples N      Number of benchmark samples (default: 7)
//   --min-time-ms T  Minimum benchmark time per problem in ms (default: 250)
//   --max-spread-pct P
//                    Fail a benchmark point when (max-min)/median exceeds P
//   --group <name>   Filter builtin sizes by group (all,tile,pack,small,workload,square,cache,aspect)
//   --size M K N     Test a single custom size
//   --threads N      Number of threads (default: 1)
//   --help           Show usage
//
// Build from the Eigen repository root (Clang 17 on Kunpeng 950). Join each
// indented command's lines with spaces when running it:
//
//   NEON:
//     clang++ -std=c++14 -O3 -DNDEBUG -flto -pthread
//       -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64
//       -march=armv9-a+nosve -I.
//       bench/sgemm_test_suite/comprehensive_tensor_contraction_bench.cpp
//       -o build-sgemm/comprehensive_tensor_contraction_bench_neon
//
//   KGemm (TensorContraction kernel specialization):
//     clang++ -std=c++14 -O3 -DNDEBUG -flto -pthread
//       -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64
//       -DEIGEN_NEON_USE_KGEMM=1 -march=armv9-a+nosve -I.
//       bench/sgemm_test_suite/comprehensive_tensor_contraction_bench.cpp
//       -o build-sgemm/comprehensive_tensor_contraction_bench_kgemm
//
//   KGemm path-instrumented verification build (do not use for performance):
//     add -DEIGEN_BENCHMARK_KGEMM_INSTRUMENTATION=1 to the KGemm command.
//
//   SVE2 (256-bit optimized Eigen GEBP kernel):
//     clang++ -std=c++14 -O3 -DNDEBUG -flto -pthread
//       -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64
//       -DEIGEN_ARM64_USE_SVE=1 -DEIGEN_BENCHMARK_REQUIRE_SVE2=1
//       -march=armv9-a+sve2 -msve-vector-bits=256 -I.
//       bench/sgemm_test_suite/comprehensive_tensor_contraction_bench.cpp
//       -o build-sgemm/comprehensive_tensor_contraction_bench_sve2
//
// Both commands use the Armv9-A base ISA. The NEON build explicitly disables
// SVE so backend comparisons do not also compare Armv8-A against Armv9-A.
//
// EIGEN_MAX_ALIGN_BYTES=64 applies the same dynamic-allocation alignment to
// both binaries. EIGEN_USE_THREADS enables ThreadPoolDevice, but the benchmark
// still uses DefaultDevice when --threads=1, so the serial hot path does not
// pay thread-pool construction or scheduling overhead. All translation units
// linked into either binary must use the same EIGEN_MAX_ALIGN_BYTES and SVE
// GEBP remainder-layout macro values.
//
// Example CPU pinning (keep all selected CPUs on the same NUMA node):
//   taskset -c 120 ./build-sgemm/comprehensive_tensor_contraction_bench_sve2 --threads 1 ...
//   taskset -c 120-123 ./build-sgemm/comprehensive_tensor_contraction_bench_sve2 --threads 4 ...
//
// TensorContraction normally packs panels and invokes Eigen Core's GEBP kernel.
// In the KGemm build, canonical contiguous FP32 contractions specialize
// TensorContractionKernel. Large contractions use the scheduler's shared block
// buffers to pack A/B once and invoke kgemm_neon_fp32_nn_packed; smaller ones
// retain raw block pointers and let KGemm own its cache blocking. Define
// EIGEN_NEON_KGEMM_REUSE_PACKING=0 for the no-reuse A/B baseline. DefaultDevice
// uses one complete GEMM block; ThreadPoolDevice retains Eigen's native
// contraction scheduling and invokes the same kernel on worker threads.
// RowMajor operands are normalized by Eigen's existing operand swap; both
// layouts are passed to KGemm as the row-major transpose C^T = B^T * A^T.
// Non-contiguous expressions and other scalar types retain the normal
// TensorContraction implementation. Effective N==1 retains Eigen's GEMV path.
//
// In the SVE2 build, selected canonical single-thread contractions with small
// internal effective M bypass packing and reuse the SVE no-pack kernel. The
// policy uses a K-vector kernel for M=1..3, an N-vector kernel for M=4..7,
// the shallow or narrow no-pack path for M=8, depth-gated M=10/M=11 paths,
// and the original no-pack kernel for M=12..15. The experimental M=8
// dual-accumulator variant is disabled by default after paired A/B testing.
// Define EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM=0 to build an A/B baseline,
// or EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM_MAX_ROWS=N to tune the upper gate.
// The packed 4..7-row remainder uses an NR=8 consumer and a K-major packed-A
// tail by default; define EIGEN_SVE_GEBP_REMAINDER_NR8=0 and/or
// EIGEN_SVE_GEBP_INTERLEAVED_REMAINDER=0 for their respective A/B baselines.
// Individual Core variants can be disabled with
// EIGEN_SVE_SMALL_GEMM_K_VECTOR_MAX_ROWS=0,
// EIGEN_SVE_SMALL_GEMM_N_VECTOR_MAX_ROWS=0, or
// EIGEN_SVE_SMALL_GEMM_M8_DUAL_ACCUMULATORS=0.
//
// For RowMajor tensors Eigen evaluates the effective GEMM as (N,K,M), so an
// external M sweep changes the internal N rather than the GEBP row count. The
// benchmark metadata records this mapping. Cases with effective N==1 use the
// Tensor GEMV path and are identified separately by verification output.
// Verification covers owning contiguous tensors with alpha=1. Strided,
// padded, deliberately misaligned buffers and the raw GEBP interface are
// outside this TensorContraction benchmark's scope.

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS 1
#endif

#if defined(EIGEN_BENCHMARK_KGEMM_INSTRUMENTATION) && \
    EIGEN_BENCHMARK_KGEMM_INSTRUMENTATION && \
    !defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION)
#define EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION 1
#endif

#include <unsupported/Eigen/CXX11/Tensor>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#if defined(EIGEN_BENCHMARK_REQUIRE_SVE2)
#if !defined(__ARM_FEATURE_SVE2)
#error "EIGEN_BENCHMARK_REQUIRE_SVE2 requires compiler support for SVE2."
#endif
#if !defined(EIGEN_VECTORIZE_SVE)
#error "EIGEN_BENCHMARK_REQUIRE_SVE2 requires Eigen's SVE backend."
#endif
#if !defined(EIGEN_ARM64_SVE_VL) || EIGEN_ARM64_SVE_VL != 256
#error "The optimized SVE2 benchmark requires a fixed 256-bit SVE vector length."
#endif
#endif

namespace {

// =========================================================================
// Types and configuration
// =========================================================================

using Clock = std::chrono::steady_clock;

struct ProblemSize {
  const char* group;
  int m;
  int k;
  int n;
};

struct BenchConfig {
  bool verify = false;             // --verify
  bool accumulate = false;         // C+=A*B instead of C=A*B
  bool verify_both_modes = false;  // --mode both
  std::string csv_file;            // --csv <path>
  std::string layout_str = "row";  // --layout row|col
  std::string kernel_str;          // --kernel neon|kgemm|sve
  std::string group = "all";       // --group
  int samples = 7;                 // --samples
  double min_time_seconds = 0.25;  // --min-time-ms
  double max_spread_pct = 0.0;     // --max-spread-pct; 0 disables failure
  int threads = 1;                 // --threads
  bool has_custom_size = false;
  int custom_m = 0, custom_k = 0, custom_n = 0;
};

volatile float benchmark_sink = 0.0f;

#if defined(EIGEN_BENCHMARK_REQUIRE_SVE2)
static_assert(Eigen::internal::packet_traits<float>::size == 8,
              "The optimized SVE2 benchmark expects eight FP32 lanes.");
static_assert(Eigen::internal::gebp_traits<float, float>::mr == 24,
              "The optimized SVE2 benchmark expects a 24-row GEBP tile.");
static_assert(Eigen::internal::gebp_traits<float, float>::nr == 4,
              "The optimized SVE2 benchmark expects a 4-column GEBP tile.");
#endif

// =========================================================================
// Vector backend detection (compile-time)
// =========================================================================

const char* vectorBackend() {
#if defined(EIGEN_VECTORIZE_SVE)
  return "SVE";
#elif defined(EIGEN_VECTORIZE_NEON)
  return "NEON";
#elif defined(EIGEN_VECTORIZE_AVX512)
  return "AVX512";
#elif defined(EIGEN_VECTORIZE_AVX)
  return "AVX";
#elif defined(EIGEN_VECTORIZE_SSE)
  return "SSE";
#else
  return "scalar";
#endif
}

int vectorBits() {
#if defined(EIGEN_ARM64_SVE_VL)
  return static_cast<int>(EIGEN_ARM64_SVE_VL);
#elif defined(EIGEN_VECTORIZE_SVE)
  return 0;  // VLA (Vector Length Agnostic) mode — bit width unknown
#else
  return static_cast<int>(sizeof(float) * 8 * Eigen::internal::packet_traits<float>::size);
#endif
}

bool optimizedSve2GebpEnabled() {
#if defined(EIGEN_VECTORIZE_SVE) && defined(__ARM_FEATURE_SVE2) && defined(EIGEN_ARM64_SVE_VL) && \
    EIGEN_ARM64_SVE_VL == 256
  using Traits = Eigen::internal::gebp_traits<float, float>;
  return Eigen::internal::packet_traits<float>::size == 8 && Traits::mr == 24 && Traits::nr == 4;
#else
  return false;
#endif
}

bool kgemmTensorKernelEnabled() {
#if defined(EIGEN_VECTORIZE_NEON) && defined(EIGEN_ARCH_ARM64) && EIGEN_ARCH_ARM64 && \
    defined(EIGEN_NEON_USE_KGEMM) && EIGEN_NEON_USE_KGEMM
  return true;
#else
  return false;
#endif
}

bool kgemmPackingReuseEnabled() {
#if defined(EIGEN_NEON_KGEMM_REUSE_PACKING) && EIGEN_NEON_KGEMM_REUSE_PACKING
  return true;
#else
  return false;
#endif
}

bool kgemmInstrumentationEnabled() {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION && defined(EIGEN_NEON_USE_KGEMM) && \
    EIGEN_NEON_USE_KGEMM
  return true;
#else
  return false;
#endif
}

bool kgemmPackReuseExpected(int m, int k, int n, int layout) {
  if (!kgemmTensorKernelEnabled() || !kgemmPackingReuseEnabled()) return false;
  const bool uses_gemv = layout == Eigen::RowMajor ? m == 1 : n == 1;
  if (uses_gemv) return false;
#if defined(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN) && \
    defined(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K)
  return std::min(m, n) >= EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN &&
         k >= EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K;
#else
  return false;
#endif
}

const char* expectedKernelPath(int m, int k, int n, int layout) {
  const bool uses_gemv = layout == Eigen::RowMajor ? m == 1 : n == 1;
  if (uses_gemv) return "gemv";
  if (!kgemmTensorKernelEnabled()) return "gebp";
  return kgemmPackReuseExpected(m, k, n, layout) ? "kgemm_packed"
                                                  : "kgemm_raw";
}

void resetKgemmInvocationCounters() {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION && defined(EIGEN_NEON_USE_KGEMM) && \
    EIGEN_NEON_USE_KGEMM
  Eigen::internal::kgemmTestResetInvocationCounters();
#endif
}

unsigned long long kgemmRawInvocationCount() {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION && defined(EIGEN_NEON_USE_KGEMM) && \
    EIGEN_NEON_USE_KGEMM
  return Eigen::internal::kgemmTestRawInvocationCount();
#else
  return 0;
#endif
}

unsigned long long kgemmPackedInvocationCount() {
#if defined(EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION) && \
    EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION && defined(EIGEN_NEON_USE_KGEMM) && \
    EIGEN_NEON_USE_KGEMM
  return Eigen::internal::kgemmTestPackedInvocationCount();
#else
  return 0;
#endif
}

bool tensorNoPackSmallGemmEnabled() {
#if defined(EIGEN_VECTORIZE_SVE) && defined(EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM) && \
    EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM && defined(EIGEN_ARM64_SVE_VL) && EIGEN_ARM64_SVE_VL == 256
  return true;
#else
  return false;
#endif
}

// =========================================================================
// CLI: argument parsing
// =========================================================================

bool parsePositiveInt(const char* text, int& value) {
  char* end = nullptr;
  long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) return false;
  value = static_cast<int>(parsed);
  return true;
}

bool parsePositiveDouble(const char* text, double& value) {
  char* end = nullptr;
  double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) return false;
  value = parsed;
  return true;
}

bool parseNonNegativeDouble(const char* text, double& value) {
  char* end = nullptr;
  double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0) return false;
  value = parsed;
  return true;
}

bool validateProblemSize(int m, int k, int n, std::string& reason) {
  if (m <= 0 || k <= 0 || n <= 0) {
    reason = "dimensions must be positive";
    return false;
  }

  const std::uint64_t index_limit = static_cast<std::uint64_t>(std::numeric_limits<Eigen::Index>::max());
  const std::uint64_t allocation_limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float));
  const std::uint64_t element_limit = std::min(index_limit, allocation_limit);
  const auto productFits = [](int rows, int cols) {
    const std::uint64_t lhs = static_cast<std::uint64_t>(rows);
    const std::uint64_t rhs = static_cast<std::uint64_t>(cols);
    return lhs <= element_limit / rhs;
  };

  if (!productFits(m, k) || !productFits(k, n) || !productFits(m, n)) {
    reason = "matrix element count exceeds Eigen::Index or addressable storage";
    return false;
  }
  const std::uint64_t mk = static_cast<std::uint64_t>(m) * k;
  const std::uint64_t kn = static_cast<std::uint64_t>(k) * n;
  const std::uint64_t mn = static_cast<std::uint64_t>(m) * n;
  if (mk > allocation_limit - kn || mk + kn > allocation_limit - mn ||
      mk + kn + mn > allocation_limit - mn) {
    reason = "combined A/B/C/reference storage exceeds addressable memory";
    return false;
  }
  return true;
}

void printUsage(const char* prog) {
  std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
            << "Options:\n"
            << "  --verify              Run correctness test (compare with naive GEMM)\n"
            << "  --mode ceqab          C = A * B  (benchmark default)\n"
            << "  --mode cpeqab         Tensor expression C += A * B\n"
            << "  --mode both           Verify ceqab and cpeqab (verification only)\n"
            << "  --csv <file>          Read m,k,n list from CSV file\n"
            << "  --layout row          RowMajor tensor storage (benchmark default)\n"
            << "  --layout col          ColMajor tensor storage\n"
            << "  --layout both         Verify both layouts (verification only)\n"
            << "  --kernel neon|kgemm|sve\n"
            << "                        Validate stock NEON, KGemm NEON, or optimized SVE2 GEBP\n"
            << "  --samples N           Number of benchmark samples (default: 7)\n"
            << "  --min-time-ms T       Minimum time per problem in ms (default: 250)\n"
            << "  --max-spread-pct P    Fail when sample spread exceeds P percent (default: disabled)\n"
            << "  --group all|tile|pack|small|workload|square|cache|aspect\n"
            << "  --size M K N          Single custom problem size\n"
            << "  --threads N           Number of threads (default: 1)\n"
            << "  --help                Show this message\n"
            << "\n"
            << "Compile-time kernel selection:\n"
            << "  NEON:  clang++ ... -O3 -DNDEBUG -pthread -DEIGEN_USE_THREADS=1"
               " -DEIGEN_MAX_ALIGN_BYTES=64 -march=armv9-a+nosve\n"
            << "  KGemm: clang++ ... -O3 -DNDEBUG -pthread -DEIGEN_USE_THREADS=1"
               " -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_NEON_USE_KGEMM=1 -march=armv9-a+nosve\n"
            << "  SVE2:  clang++ ... -O3 -DNDEBUG -pthread -DEIGEN_USE_THREADS=1"
               " -DEIGEN_MAX_ALIGN_BYTES=64 -DEIGEN_ARM64_USE_SVE=1"
               " -DEIGEN_BENCHMARK_REQUIRE_SVE2=1 -march=armv9-a+sve2 -msve-vector-bits=256\n";
}

bool parseOptions(int argc, char** argv, BenchConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--verify") {
      cfg.verify = true;
    } else if (arg == "--mode" && i + 1 < argc) {
      std::string mode(argv[++i]);
      if (mode == "ceqab") {
        cfg.accumulate = false;
        cfg.verify_both_modes = false;
      } else if (mode == "cpeqab") {
        cfg.accumulate = true;
        cfg.verify_both_modes = false;
      } else if (mode == "both") {
        cfg.verify_both_modes = true;
      } else {
        std::cerr << "Invalid mode: " << mode << " (use ceqab, cpeqab, or both)\n";
        return false;
      }
    } else if (arg == "--csv" && i + 1 < argc) {
      cfg.csv_file = argv[++i];
    } else if (arg == "--layout" && i + 1 < argc) {
      std::string layout(argv[++i]);
      if (layout == "row" || layout == "col" || layout == "both")
        cfg.layout_str = layout;
      else {
        std::cerr << "Invalid layout: " << layout << " (use row, col, or both)\n";
        return false;
      }
    } else if (arg == "--kernel" && i + 1 < argc) {
      std::string kernel(argv[++i]);
      if (kernel == "neon" || kernel == "kgemm" || kernel == "sve")
        cfg.kernel_str = kernel;
      else {
        std::cerr << "Invalid kernel: " << kernel << " (use neon, kgemm, or sve)\n";
        return false;
      }
    } else if (arg == "--samples" && i + 1 < argc) {
      if (!parsePositiveInt(argv[++i], cfg.samples)) {
        std::cerr << "Invalid --samples\n";
        return false;
      }
    } else if (arg == "--min-time-ms" && i + 1 < argc) {
      double ms = 0.0;
      if (!parsePositiveDouble(argv[++i], ms)) {
        std::cerr << "Invalid --min-time-ms\n";
        return false;
      }
      cfg.min_time_seconds = ms / 1000.0;
    } else if (arg == "--max-spread-pct" && i + 1 < argc) {
      if (!parseNonNegativeDouble(argv[++i], cfg.max_spread_pct)) {
        std::cerr << "Invalid --max-spread-pct\n";
        return false;
      }
    } else if (arg == "--group" && i + 1 < argc) {
      cfg.group = argv[++i];
      if (cfg.group != "all" && cfg.group != "tile" && cfg.group != "pack" && cfg.group != "small" &&
          cfg.group != "workload" && cfg.group != "square" && cfg.group != "cache" && cfg.group != "aspect" &&
          cfg.group != "csv" && cfg.group != "custom") {
        std::cerr << "Invalid group: " << cfg.group << "\n";
        return false;
      }
    } else if (arg == "--size" && i + 3 < argc) {
      int m = 0, k = 0, n = 0;
      if (!parsePositiveInt(argv[++i], m) || !parsePositiveInt(argv[++i], k) || !parsePositiveInt(argv[++i], n)) {
        std::cerr << "Invalid --size M K N\n";
        return false;
      }
      cfg.custom_m = m;
      cfg.custom_k = k;
      cfg.custom_n = n;
      cfg.has_custom_size = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      if (!parsePositiveInt(argv[++i], cfg.threads)) {
        std::cerr << "Invalid --threads\n";
        return false;
      }
    } else {
      std::cerr << "Unknown argument: " << arg << " (use --help)\n";
      return false;
    }
  }
  if (cfg.has_custom_size && !cfg.csv_file.empty()) {
    std::cerr << "--size and --csv are mutually exclusive\n";
    return false;
  }
  if (!cfg.verify && cfg.verify_both_modes) {
    std::cerr << "--mode both is only valid with --verify\n";
    return false;
  }
  if (!cfg.verify && cfg.layout_str == "both") {
    std::cerr << "--layout both is only valid with --verify\n";
    return false;
  }
  if (cfg.has_custom_size && cfg.group != "all" && cfg.group != "custom") {
    std::cerr << "--size may only be combined with --group all or custom\n";
    return false;
  }
  if (!cfg.csv_file.empty() && cfg.group != "all" && cfg.group != "csv") {
    std::cerr << "--csv may only be combined with --group all or csv\n";
    return false;
  }
  if (!cfg.has_custom_size && cfg.csv_file.empty() && (cfg.group == "custom" || cfg.group == "csv")) {
    std::cerr << "--group " << cfg.group << " requires --" << cfg.group << "\n";
    return false;
  }
  if (cfg.has_custom_size) {
    std::string reason;
    if (!validateProblemSize(cfg.custom_m, cfg.custom_k, cfg.custom_n, reason)) {
      std::cerr << "Invalid --size M K N: " << reason << "\n";
      return false;
    }
  }
  return true;
}

// =========================================================================
// CSV reading
// =========================================================================

std::string trim(const std::string& value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return std::string();
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

bool splitCsvRecord(const std::string& line, std::string& first, std::string& second, std::string& third) {
  const std::size_t first_comma = line.find(',');
  if (first_comma == std::string::npos) return false;
  const std::size_t second_comma = line.find(',', first_comma + 1);
  if (second_comma == std::string::npos || line.find(',', second_comma + 1) != std::string::npos) return false;

  first = trim(line.substr(0, first_comma));
  second = trim(line.substr(first_comma + 1, second_comma - first_comma - 1));
  third = trim(line.substr(second_comma + 1));
  return !first.empty() && !second.empty() && !third.empty();
}

bool readCsv(const std::string& path, std::vector<ProblemSize>& sizes) {
  sizes.clear();
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error: cannot open CSV file: " << path << "\n";
    return false;
  }

  std::string line;
  // Skip header line
  if (!std::getline(file, line)) {
    std::cerr << "Error: CSV file is empty: " << path << "\n";
    return false;
  }
  if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xef && static_cast<unsigned char>(line[1]) == 0xbb &&
      static_cast<unsigned char>(line[2]) == 0xbf) {
    line.erase(0, 3);
  }
  std::string h_m, h_k, h_n;
  if (!splitCsvRecord(line, h_m, h_k, h_n) || h_m != "m" || h_k != "k" || h_n != "n") {
    std::cerr << "Error: CSV header must contain exactly: m,k,n\n";
    return false;
  }

  int lineno = 1;
  while (std::getline(file, line)) {
    ++lineno;
    const std::size_t first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;

    const std::string original_line = line;
    std::string field_m, field_k, field_n;
    int m = 0, k = 0, n = 0;
    if (!splitCsvRecord(line, field_m, field_k, field_n) || !parsePositiveInt(field_m.c_str(), m) ||
        !parsePositiveInt(field_k.c_str(), k) || !parsePositiveInt(field_n.c_str(), n)) {
      std::cerr << "Error: invalid m,k,n at line " << lineno << " in " << path << ": " << original_line << "\n";
      return false;
    }
    std::string reason;
    if (!validateProblemSize(m, k, n, reason)) {
      std::cerr << "Error: invalid dimensions at line " << lineno << " in " << path << ": " << reason << "\n";
      return false;
    }
    sizes.push_back({"csv", m, k, n});
  }
  if (sizes.empty()) {
    std::cerr << "Error: CSV contains no problem sizes: " << path << "\n";
    return false;
  }
  return true;
}

// =========================================================================
// Builtin problem sizes (comprehensive coverage)
// =========================================================================

const ProblemSize kBuiltinSizes[] = {
    // ---- tile: boundary conditions around register-block sizes ----
    {"tile", 239, 256, 256},
    {"tile", 240, 256, 256},
    {"tile", 241, 256, 256},
    {"tile", 240, 255, 256},
    {"tile", 240, 257, 256},
    {"tile", 240, 256, 255},
    {"tile", 240, 256, 257},

    // ---- pack: reuse gate and packed-panel tail conditions ----
    {"pack", 767, 512, 767},
    {"pack", 768, 511, 768},
    {"pack", 768, 512, 768},
    {"pack", 769, 512, 769},
    {"pack", 781, 513, 783},

    // ---- small: varying M with fixed K=N=400 ----
    {"small", 8, 400, 400},
    {"small", 14, 400, 400},
    {"small", 16, 400, 400},
    {"small", 23, 400, 400},
    {"small", 24, 400, 400},
    {"small", 31, 400, 400},
    {"small", 32, 400, 400},
    {"small", 35, 400, 400},
    {"small", 39, 400, 400},
    {"small", 40, 400, 400},
    {"small", 41, 400, 400},
    {"small", 47, 400, 400},
    {"small", 48, 400, 400},
    {"small", 50, 400, 400},
    {"small", 55, 400, 400},
    {"small", 59, 400, 400},
    {"small", 63, 400, 400},
    {"small", 64, 400, 400},
    {"small", 71, 400, 400},
    {"small", 72, 400, 400},
    {"small", 77, 400, 400},
    {"small", 80, 400, 400},
    {"small", 81, 400, 400},
    {"small", 88, 400, 400},
    {"small", 95, 400, 400},
    {"small", 96, 400, 400},

    // ---- workload: sizes from sgemm_test_plan.md ----
    {"workload", 14, 28, 16},
    {"workload", 16, 8, 4},
    {"workload", 19, 8, 4},
    {"workload", 1, 16, 16},
    {"workload", 1, 24, 16},
    {"workload", 32, 16, 8},
    {"workload", 32, 32, 8},
    {"workload", 32, 8, 4},
    {"workload", 35, 400, 400},
    {"workload", 35, 800, 1},
    {"workload", 39, 400, 400},
    {"workload", 39, 492, 1},
    {"workload", 39, 5, 1},
    {"workload", 39, 800, 1},
    {"workload", 4130, 104, 32},
    {"workload", 4130, 116, 32},
    {"workload", 4130, 16, 1},
    {"workload", 4130, 32, 16},
    {"workload", 46, 492, 1},
    {"workload", 46, 5, 1},
    {"workload", 48, 18, 16},
    {"workload", 50, 100, 1},
    {"workload", 50, 1298, 250},
    {"workload", 50, 250, 100},
    {"workload", 50, 250, 250},
    {"workload", 59, 128, 64},
    {"workload", 59, 1316, 2},
    {"workload", 59, 1316, 256},
    {"workload", 59, 1540, 64},
    {"workload", 59, 1570, 128},
    {"workload", 59, 1570, 256},
    {"workload", 59, 16, 8},
    {"workload", 59, 256, 128},
    {"workload", 59, 30, 16},
    {"workload", 59, 32, 2},
    {"workload", 59, 48, 48},
    {"workload", 59, 48, 80},
    {"workload", 59, 64, 1},
    {"workload", 59, 64, 2},
    {"workload", 59, 64, 32},
    {"workload", 59, 64, 6},
    {"workload", 59, 8, 1},
    {"workload", 77, 20, 16},
    {"workload", 80, 36, 16},

    // ---- square: small to large square matrices ----
    {"square", 96, 96, 96},
    {"square", 128, 128, 128},
    {"square", 192, 192, 192},
    {"square", 256, 256, 256},
    {"square", 384, 384, 384},
    {"square", 512, 512, 512},
    {"square", 768, 768, 768},
    {"square", 1024, 1024, 1024},
    {"square", 1536, 1536, 1536},
    {"square", 2048, 2048, 2048},
    {"square", 4096, 4096, 4096},

    // ---- cache: large M with square K,N ----
    {"cache", 4096, 96, 96},
    {"cache", 4096, 128, 128},
    {"cache", 4096, 144, 144},

    // ---- aspect: high-aspect-ratio matrices ----
    {"aspect", 128, 4096, 128},
    {"aspect", 128, 128, 4096},
};

int numBuiltinSizes() { return sizeof(kBuiltinSizes) / sizeof(kBuiltinSizes[0]); }

// The verification set is intentionally smaller than the full benchmark set,
// but it covers the SVE 24-row tile, every 3-8-row remainder around that tile,
// the 255/256/257 panel boundaries, shallow K, N=1/GEMV workloads, and a few
// representative cache/aspect shapes. Square tail cases exercise the same
// effective GEMM M for RowMajor and ColMajor despite RowMajor's M/N swap.
const ProblemSize kVerificationSizes[] = {
    // Full-tile and packing boundaries from the benchmark set.
    {"tile", 239, 256, 256},
    {"tile", 240, 256, 256},
    {"tile", 241, 256, 256},
    {"tile", 240, 255, 256},
    {"tile", 240, 257, 256},
    {"tile", 240, 256, 255},
    {"tile", 240, 256, 257},

    // Pack-reuse gate and packed A/B/output tails.
    {"pack", 767, 512, 767},
    {"pack", 768, 511, 768},
    {"pack", 768, 512, 768},
    {"pack", 769, 512, 769},
    {"pack", 781, 513, 783},

    // 24-row boundary and 3-8 residual rows; K is deliberately non-aligned.
    {"tile", 23, 37, 23},
    {"tile", 24, 37, 24},
    {"tile", 25, 37, 25},
    {"tile", 27, 37, 27},
    {"tile", 28, 37, 28},
    {"tile", 29, 37, 29},
    {"tile", 30, 37, 30},
    {"tile", 31, 37, 31},
    {"tile", 32, 37, 32},

    // Shallow-K paths, including K=1/2/6.
    {"tile", 31, 1, 29},
    {"tile", 31, 2, 29},
    {"tile", 31, 6, 29},

    {"small", 8, 400, 400},
    {"small", 32, 400, 400},
    {"small", 64, 400, 400},

    {"workload", 1, 16, 16},
    {"workload", 14, 28, 16},
    {"workload", 32, 32, 8},
    {"workload", 48, 18, 16},
    {"workload", 59, 64, 32},
    {"workload", 80, 36, 16},
    {"workload", 35, 800, 1},
    {"workload", 39, 5, 1},
    {"workload", 59, 64, 1},

    {"square", 96, 96, 96},
    {"square", 128, 128, 128},
    {"cache", 4096, 96, 96},
    {"aspect", 128, 4096, 128},
    {"aspect", 128, 128, 4096},
};

int numVerificationSizes() { return sizeof(kVerificationSizes) / sizeof(kVerificationSizes[0]); }

// =========================================================================
// Deterministic inputs and reusable contraction runner
// =========================================================================

constexpr std::uint64_t kSeedA = UINT64_C(0x243f6a8885a308d3);
constexpr std::uint64_t kSeedB = UINT64_C(0x13198a2e03707344);
constexpr std::uint64_t kSeedC = UINT64_C(0xa4093822299f31d0);

float deterministicValue(Eigen::Index row, Eigen::Index col, std::uint64_t seed) {
  std::uint64_t value = seed ^ (static_cast<std::uint64_t>(row) + 1) * UINT64_C(0x9e3779b97f4a7c15) ^
                        (static_cast<std::uint64_t>(col) + 1) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31;
  std::uint32_t bits = UINT32_C(0x3f000000) | static_cast<std::uint32_t>(value & UINT64_C(0x007fffff));
  if ((value & UINT64_C(0x8000000000000000)) != 0) bits |= UINT32_C(0x80000000);
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;  // A normal, nonzero value in [-1,-0.5] or [0.5,1].
}

template <typename Tensor2D>
void fillDeterministic(Tensor2D& tensor, int rows, int cols, std::uint64_t seed) {
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index col = 0; col < cols; ++col) tensor(row, col) = deterministicValue(row, col, seed);
  }
}

