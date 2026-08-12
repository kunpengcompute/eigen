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

template <int Layout>
Eigen::Index tensorOffset(Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols) {
  return Layout == Eigen::RowMajor ? row * cols + col : row + col * rows;
}

EIGEN_STRONG_INLINE void compilerMemoryBarrier(const void* pointer) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("" : : "g"(pointer) : "memory");
#else
  EIGEN_UNUSED_VARIABLE(pointer);
#endif
}

template <int Layout, typename Device, bool Accumulate>
class ContractionBenchmark {
 public:
  using Tensor2D = Eigen::Tensor<float, 2, Layout>;

  ContractionBenchmark(int m, int k, int n, const Device& device)
      : m_(m),
        n_(n),
        device_(device),
        a_(m, k),
        b_(k, n),
        c_(m, n),
        initial_c_(static_cast<std::size_t>(m) * static_cast<std::size_t>(n)) {
    dims_[0] = Eigen::IndexPair<int>(1, 0);
    fillDeterministic(a_, m, k, kSeedA);
    fillDeterministic(b_, k, n, kSeedB);
    fillDeterministic(c_, m, n, kSeedC);
    std::copy(c_.data(), c_.data() + initial_c_.size(), initial_c_.begin());
  }

  EIGEN_DONT_INLINE double run(std::int64_t iters) {
    resetOutput();
    execute();  // Warm the same tensors and code path used by the timed loop.

    // Preserve the cache and worker-thread state produced by the warm-up.  A
    // second host-side copy of C here would evict parts of A/B for large
    // problems and pull output cache lines back to the caller thread.  C=A*B
    // overwrites this value; C+=A*B remains the same operation with one
    // untimed accumulation already applied.

    const auto start = Clock::now();
    for (std::int64_t i = 0; i < iters; ++i) {
      execute();
      compilerMemoryBarrier(c_.data());
    }
    const auto stop = Clock::now();
    benchmark_sink += c_(0, 0);
    return std::chrono::duration<double>(stop - start).count();
  }

  double oneOperationChecksum() {
    resetOutput();
    execute();
    double checksum = 0.0;
    const Eigen::Index count = static_cast<Eigen::Index>(m_) * static_cast<Eigen::Index>(n_);
    for (Eigen::Index i = 0; i < count; ++i) checksum += static_cast<double>(c_.data()[i]);
    benchmark_sink += c_(0, 0);
    return checksum;
  }

 private:
  void resetOutput() { std::copy(initial_c_.begin(), initial_c_.end(), c_.data()); }

  EIGEN_ALWAYS_INLINE void executeImpl(std::true_type) {
    c_.device(device_) += a_.contract(b_, dims_);
  }

  EIGEN_ALWAYS_INLINE void executeImpl(std::false_type) {
    c_.device(device_) = a_.contract(b_, dims_);
  }

  EIGEN_ALWAYS_INLINE void execute() {
    executeImpl(std::integral_constant<bool, Accumulate>{});
  }

  int m_;
  int n_;
  const Device& device_;
  Tensor2D a_;
  Tensor2D b_;
  Tensor2D c_;
  std::vector<float> initial_c_;
  Eigen::array<Eigen::IndexPair<int>, 1> dims_;
};

// Calibrate: find number of iterations needed to reach min_time.
template <typename Runner>
std::int64_t calibrateIters(Runner& runner, double min_time) {
  const double calib_time = std::min(0.05, min_time / 4.0);
  const std::int64_t max_iters = std::numeric_limits<std::int32_t>::max();
  std::int64_t iters = 1;
  double elapsed = runner.run(iters);

  while (elapsed < calib_time && iters <= max_iters / 2) {
    iters *= 2;
    elapsed = runner.run(iters);
  }

  if (!std::isfinite(elapsed) || elapsed <= 0.0) return iters;
  const double scale = min_time / elapsed;
  const double scaled = std::ceil(static_cast<double>(iters) * scale);
  if (!std::isfinite(scaled) || scaled >= static_cast<double>(max_iters)) return max_iters;
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(scaled));
}

// =========================================================================
// Correctness verification
// =========================================================================

float exactBValue(Eigen::Index row, Eigen::Index col) {
  const std::uint64_t mixed = static_cast<std::uint64_t>(row) * UINT64_C(17) +
                              static_cast<std::uint64_t>(col) * UINT64_C(13);
  return static_cast<float>(static_cast<int>(mixed % UINT64_C(7)) - 3);
}

float exactCValue(Eigen::Index row, Eigen::Index col) {
  const std::uint64_t mixed = static_cast<std::uint64_t>(row) * UINT64_C(5) +
                              static_cast<std::uint64_t>(col) * UINT64_C(11);
  return static_cast<float>(static_cast<int>(mixed % UINT64_C(5)) - 2);
}

template <typename Tensor2D>
void fillExactOneHotInputs(Tensor2D& a, Tensor2D& b, Tensor2D& c,
                           int m, int k, int n) {
  a.setZero();
  for (Eigen::Index row = 0; row < m; ++row) {
    const Eigen::Index pivot =
        (row * static_cast<Eigen::Index>(131) + 7) % k;
    a(row, pivot) = 1.0f;
  }
  for (Eigen::Index row = 0; row < k; ++row) {
    for (Eigen::Index col = 0; col < n; ++col) {
      b(row, col) = exactBValue(row, col);
    }
  }
  for (Eigen::Index row = 0; row < m; ++row) {
    for (Eigen::Index col = 0; col < n; ++col) {
      c(row, col) = exactCValue(row, col);
    }
  }
}

template <int Layout, typename Device>
bool verifyContraction(int m, int k, int n, bool accumulate, const Device& device) {
  using Tensor2D = Eigen::Tensor<float, 2, Layout>;

  Tensor2D A(m, k);
  Tensor2D B(k, n);
  Tensor2D C_eigen(m, n);

  fillDeterministic(A, m, k, kSeedA);
  fillDeterministic(B, k, n, kSeedB);
  fillDeterministic(C_eigen, m, n, kSeedC);
  const Eigen::Index output_count = static_cast<Eigen::Index>(m) * static_cast<Eigen::Index>(n);
  std::vector<float> initial_c(C_eigen.data(), C_eigen.data() + output_count);

  // Eigen contraction
  Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(1, 0)};
  resetKgemmInvocationCounters();
  if (accumulate) {
    C_eigen.device(device) += A.contract(B, dims);
  } else {
    C_eigen.device(device) = A.contract(B, dims);
  }

  // Compare against a double-precision dot product. gamma_(2K+1) bounds the
  // FP32 multiply/add rounding; a safety factor covers vectorized reduction
  // order, FMA/non-FMA differences, and the separate Tensor += expression.
  const double unit_roundoff = static_cast<double>(std::numeric_limits<float>::epsilon()) / 2.0;
  const double gamma_argument = (2.0 * static_cast<double>(k) + 1.0) * unit_roundoff;
  if (gamma_argument >= 1.0) {
    std::cerr << "Cannot construct a meaningful FP32 error bound for K=" << k << "\n";
    return false;
  }
  const double gamma = gamma_argument / (1.0 - gamma_argument);
  constexpr double kErrorSafetyFactor = 8.0;

  double max_rel_error = 0.0;
  double max_abs_error = 0.0;
  double max_bound_ratio = 0.0;
  double sum_rel_error = 0.0;
  std::uint64_t err_count = 0;
  std::uint64_t nonfinite_count = 0;

  for (Eigen::Index i = 0; i < m; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      const Eigen::Index c_offset = tensorOffset<Layout>(i, j, m, n);
      const double initial = static_cast<double>(initial_c[static_cast<std::size_t>(c_offset)]);
      double expected = accumulate ? initial : 0.0;
      double magnitude_sum = accumulate ? std::abs(initial) : 0.0;
      for (Eigen::Index l = 0; l < k; ++l) {
        const Eigen::Index a_offset = tensorOffset<Layout>(i, l, m, k);
        const Eigen::Index b_offset = tensorOffset<Layout>(l, j, k, n);
        const double product = static_cast<double>(A.data()[a_offset]) * static_cast<double>(B.data()[b_offset]);
        expected += product;
        magnitude_sum += std::abs(product);
      }

      const double got = static_cast<double>(C_eigen.data()[c_offset]);
      const bool finite = std::isfinite(expected) && std::isfinite(got);
      const double abs_error = finite ? std::abs(expected - got) : std::numeric_limits<double>::infinity();
      const double rel_error =
          finite ? abs_error / std::max(1.0, std::abs(expected)) : std::numeric_limits<double>::infinity();
      const double error_bound =
          kErrorSafetyFactor * (gamma * magnitude_sum + unit_roundoff * std::max(1.0, std::abs(expected)));
      const double bound_ratio =
          finite && error_bound > 0.0 ? abs_error / error_bound : std::numeric_limits<double>::infinity();
      const bool mismatch = !finite || !(abs_error <= error_bound);
      if (mismatch) {
        ++err_count;
        if (!finite) ++nonfinite_count;
        if (err_count <= 10) {  // print first 10 errors
          std::cerr << "  MISMATCH at (" << i << "," << j << "): expected=" << expected << " got=" << got
                    << " abs_err=" << abs_error << " error_bound=" << error_bound << "\n";
        }
      }
      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);
      max_bound_ratio = std::max(max_bound_ratio, bound_ratio);
      sum_rel_error += rel_error;
    }
  }

  const double total = static_cast<double>(output_count);
  const double avg_rel_error = sum_rel_error / total;

  // A one-hot A matrix and small integral B/C values have an exactly
  // representable result.  This catches packed-lane swaps, dropped K entries,
  // and tail corruption that can fit inside the conservative FP error bound.
  fillExactOneHotInputs(A, B, C_eigen, m, k, n);
  if (accumulate) {
    C_eigen.device(device) += A.contract(B, dims);
  } else {
    C_eigen.device(device) = A.contract(B, dims);
  }

  std::uint64_t exact_err_count = 0;
  double exact_max_abs_error = 0.0;
  for (Eigen::Index i = 0; i < m; ++i) {
    const Eigen::Index pivot =
        (i * static_cast<Eigen::Index>(131) + 7) % k;
    for (Eigen::Index j = 0; j < n; ++j) {
      const double expected = static_cast<double>(exactBValue(pivot, j)) +
                              (accumulate ? exactCValue(i, j) : 0.0f);
      const Eigen::Index offset = tensorOffset<Layout>(i, j, m, n);
      const double got = static_cast<double>(C_eigen.data()[offset]);
      const double abs_error = std::abs(expected - got);
      exact_max_abs_error = std::max(exact_max_abs_error, abs_error);
      if (!std::isfinite(got) || got != expected) {
        ++exact_err_count;
        if (exact_err_count <= 10) {
          std::cerr << "  EXACT MISMATCH at (" << i << ',' << j
                    << "): expected=" << expected << " got=" << got << "\n";
        }
      }
    }
  }

  const Eigen::Index effective_m = Layout == Eigen::RowMajor ? n : m;
  const Eigen::Index effective_n = Layout == Eigen::RowMajor ? m : n;
  const bool uses_gemv = effective_n == 1;
  const bool uses_kgemm =
      !uses_gemv && kgemmTensorKernelEnabled() &&
      (std::is_same<Device, Eigen::DefaultDevice>::value ||
       std::is_same<Device, Eigen::ThreadPoolDevice>::value);
  const bool expects_packed =
      kgemmPackReuseExpected(m, k, n, Layout);
  const unsigned long long raw_invocations = kgemmRawInvocationCount();
  const unsigned long long packed_invocations = kgemmPackedInvocationCount();
  bool path_ok = true;
  if (kgemmInstrumentationEnabled()) {
    if (!uses_kgemm) {
      path_ok = raw_invocations == 0 && packed_invocations == 0;
    } else if (expects_packed) {
      path_ok = packed_invocations > 0 && raw_invocations == 0;
    } else {
      path_ok = raw_invocations > 0 && packed_invocations == 0;
    }
  }
  const bool all_ok = err_count == 0 && exact_err_count == 0 && path_ok;
  const char* implementation = uses_kgemm ? "KGEMM" : (uses_gemv ? "GEMV" : "GEBP");
  std::cout << "  [" << (Layout == Eigen::RowMajor ? "RowMajor" : "ColMajor") << ',' << implementation
            << ",effective=" << effective_m << 'x' << k << 'x' << effective_n << "] "
            << (accumulate ? "C+=A*B" : "C=A*B") << " (" << m << "x" << k << "x" << n << "): "
            << "max_abs_err=" << max_abs_error << " max_rel_err=" << max_rel_error << " avg_rel_err=" << avg_rel_error
            << " max_bound_ratio=" << max_bound_ratio << " nonfinite=" << nonfinite_count << " mismatches=" << err_count
            << "/" << output_count << " exact_mismatches=" << exact_err_count
            << " exact_max_abs_err=" << exact_max_abs_error;
  if (kgemmInstrumentationEnabled()) {
    std::cout << " raw_invocations=" << raw_invocations
              << " packed_invocations=" << packed_invocations
              << " expected_path="
              << (uses_kgemm ? (expects_packed ? "kgemm_packed" : "kgemm_raw")
                             : (uses_gemv ? "gemv" : "gebp"))
              << " path_check=" << (path_ok ? "PASS" : "FAIL");
  } else {
    std::cout << " path_check=not_instrumented";
  }
  std::cout << (all_ok ? "  PASS" : "  FAIL") << "\n";
  return all_ok;
}

// =========================================================================
// Benchmark runner (layout dispatch)
// =========================================================================

template <typename Device>
bool runBenchmarkProblem(const ProblemSize& sz, const BenchConfig& cfg, const Device& device) {
  double gflops = 0.0, gflops_min = 0.0, gflops_max = 0.0;
  double spread_pct = 0.0;
  std::int64_t iters = 0;
  double checksum = 0.0;
  const char* kernel_path = "unknown";

  auto do_run = [&](auto layout_tag, auto accumulate_tag) -> bool {
    constexpr int Layout = decltype(layout_tag)::value;
    constexpr bool Accumulate = decltype(accumulate_tag)::value;
    ContractionBenchmark<Layout, Device, Accumulate> runner(sz.m, sz.k, sz.n, device);
    kernel_path = expectedKernelPath(sz.m, sz.k, sz.n, Layout);
    checksum = runner.oneOperationChecksum();
    iters = calibrateIters(runner, cfg.min_time_seconds);

    std::vector<double> gflops_samples;
    gflops_samples.reserve(cfg.samples);
    const double flops = 2.0 * static_cast<double>(sz.m) * static_cast<double>(sz.n) * static_cast<double>(sz.k);

    for (int s = 0; s < cfg.samples; ++s) {
      const double elapsed = runner.run(iters);
      if (!std::isfinite(elapsed) || elapsed <= 0.0) {
        std::cerr << "Invalid elapsed time for " << sz.m << ',' << sz.k << ',' << sz.n << "\n";
        return false;
      }
      gflops_samples.push_back(1e-9 * flops * iters / elapsed);
    }
    std::sort(gflops_samples.begin(), gflops_samples.end());

    const std::size_t middle = gflops_samples.size() / 2;
    gflops = gflops_samples.size() % 2 == 0 ? 0.5 * (gflops_samples[middle - 1] + gflops_samples[middle])
                                            : gflops_samples[middle];
    gflops_min = gflops_samples.front();
    gflops_max = gflops_samples.back();
    spread_pct = 100.0 * (gflops_max - gflops_min) / gflops;
    if (cfg.max_spread_pct > 0.0 && spread_pct > cfg.max_spread_pct) {
      std::cerr << "Sample spread " << spread_pct << "% exceeds --max-spread-pct "
                << cfg.max_spread_pct << "% for " << sz.m << ',' << sz.k << ',' << sz.n << "\n";
      return false;
    }
    return true;
  };

  // Select the runtime mode once per problem. The timed loop then enters a
  // compile-time-specialized execute path with no accumulate/overwrite branch.
  auto dispatch_mode = [&](auto layout_tag) -> bool {
    if (cfg.accumulate) {
      return do_run(layout_tag, std::true_type{});
    }
    return do_run(layout_tag, std::false_type{});
  };

  bool ok = false;
  if (cfg.layout_str == "row") {
    ok = dispatch_mode(std::integral_constant<int, Eigen::RowMajor>{});
  } else {
    ok = dispatch_mode(std::integral_constant<int, Eigen::ColMajor>{});
  }
  if (!ok) return false;

  std::cout << sz.group << ',' << sz.m << ',' << sz.k << ',' << sz.n << ',' << iters << ',' << gflops << ','
            << gflops_min << ',' << gflops_max << ',' << checksum << ',' << spread_pct << ',' << kernel_path << '\n';
  return true;
}

// =========================================================================
// Metadata output
// =========================================================================

void printMetadata(const BenchConfig& cfg, const char* tensor_device) {
  using Traits = Eigen::internal::gebp_traits<float, float>;

  std::cout << "# benchmark=comprehensive_tensor_contraction\n";
  std::cout << "# compiler=" << __VERSION__ << "\n";
  std::cout << "# vector_backend=" << vectorBackend() << "\n";
  std::cout << "# vector_bits=" << vectorBits() << "\n";
  std::cout << "# packet_floats=" << Eigen::internal::packet_traits<float>::size << "\n";
  std::cout << "# gebp_mr=" << Traits::mr << "\n";
  std::cout << "# gebp_nr=" << Traits::nr << "\n";
  std::cout << "# sve2_optimized_gebp=" << (optimizedSve2GebpEnabled() ? 1 : 0) << "\n";
  std::cout << "# kgemm_tensor_kernel=" << (kgemmTensorKernelEnabled() ? 1 : 0) << "\n";
  std::cout << "# kgemm_tensor_kernel_policy=canonical_contiguous_fp32_default_or_threadpool\n";
  std::cout << "# kgemm_tensor_kernel_expected="
            << (kgemmTensorKernelEnabled() ? 1 : 0) << "\n";
  std::cout << "# kgemm_pack_reuse=" << (kgemmPackingReuseEnabled() ? 1 : 0) << "\n";
  std::cout << "# kgemm_test_instrumentation=" << (kgemmInstrumentationEnabled() ? 1 : 0) << "\n";
#if defined(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN)
  std::cout << "# kgemm_pack_reuse_min_mn=" << EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN << "\n";
  std::cout << "# kgemm_pack_reuse_min_k=" << EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K << "\n";
#endif
  std::cout << "# dedicated_small_gemm=" << (tensorNoPackSmallGemmEnabled() ? 1 : 0) << "\n";
#if defined(EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM_MAX_ROWS)
  std::cout << "# tensor_small_gemm_max_rows=" << EIGEN_SVE_TENSOR_CONTRACTION_SMALL_GEMM_MAX_ROWS << "\n";
  std::cout << "# tensor_small_gemm_policy=m1_3_kvec_kge2,m4_7_nvec,m8_shallow_or_narrow,m10_deep,m11_deep,m12_15\n";
  std::cout << "# sve_small_gemm_k_vector_max_rows=" << EIGEN_SVE_SMALL_GEMM_K_VECTOR_MAX_ROWS << "\n";
  std::cout << "# sve_small_gemm_k_vector_min_depth=" << EIGEN_SVE_SMALL_GEMM_K_VECTOR_MIN_DEPTH << "\n";
  std::cout << "# sve_small_gemm_n_vector_max_rows=" << EIGEN_SVE_SMALL_GEMM_N_VECTOR_MAX_ROWS << "\n";
  std::cout << "# sve_small_gemm_m8_dual_accumulators=" << EIGEN_SVE_SMALL_GEMM_M8_DUAL_ACCUMULATORS << "\n";
  std::cout << "# tensor_m8_deep_max_cols=" << EIGEN_SVE_TENSOR_CONTRACTION_M8_DEEP_MAX_COLS << "\n";
  std::cout << "# tensor_m8_deep_max_depth=" << EIGEN_SVE_TENSOR_CONTRACTION_M8_DEEP_MAX_DEPTH << "\n";
  std::cout << "# tensor_m10_min_depth=" << EIGEN_SVE_TENSOR_CONTRACTION_M10_MIN_DEPTH << "\n";
  std::cout << "# tensor_m11_min_depth=" << EIGEN_SVE_TENSOR_CONTRACTION_M11_MIN_DEPTH << "\n";
#endif
#if defined(EIGEN_SVE_GEBP_REMAINDER_NR8)
  std::cout << "# sve_gebp_remainder_nr8=" << EIGEN_SVE_GEBP_REMAINDER_NR8 << "\n";
#endif
#if defined(EIGEN_SVE_GEBP_INTERLEAVED_REMAINDER)
  std::cout << "# sve_gebp_interleaved_remainder=" << EIGEN_SVE_GEBP_INTERLEAVED_REMAINDER << "\n";
#endif
  std::cout << "# tuple_order=m,k,n\n";
  std::cout << "# max_align_bytes=" << EIGEN_MAX_ALIGN_BYTES << "\n";
  std::cout << "# tensor_device=" << tensor_device << "\n";
  std::cout << "# eigen_threads=1\n";
  std::cout << "# tensor_threads=" << cfg.threads << "\n";
  std::cout << "# threads=" << cfg.threads << "\n";
  std::cout << "# layout=" << cfg.layout_str << "\n";
  if (cfg.layout_str == "row") {
    std::cout << "# effective_gemm_mapping=external(m,k,n)->internal(n,k,m)\n";
    std::cout << "# gemv_condition=external_m==1\n";
  } else {
    std::cout << "# effective_gemm_mapping=external(m,k,n)->internal(m,k,n)\n";
    std::cout << "# gemv_condition=external_n==1\n";
  }
  std::cout << "# mode=" << (cfg.accumulate ? "C+=A*B" : "C=A*B") << "\n";
  std::cout << "# cpeqab_semantics=tensor_expression_not_fused_beta1_gemm\n";
  std::cout << "# input_pattern=deterministic_normal_v2\n";
  std::cout << "# initial_c=deterministic_nonzero\n";
  std::cout << "# checksum_scope=one_operation_same_inputs\n";
  std::cout << "# cache_state=warm_steady_state\n";
  std::cout << "# warmup_operations_per_sample=1\n";
  std::cout << "# output_reset=before_warmup_outside_timing\n";
  std::cout << "# timed_loop_wrapper=noinline\n";
  std::cout << "# contraction_call=always_inline\n";
  std::cout << "# runtime_mode_branch_in_timed_loop=0\n";
  std::cout << "# tensor_scope=owning_contiguous_alpha1\n";
  std::cout << "# samples=" << cfg.samples << "\n";
  std::cout << "# min_time_ms=" << cfg.min_time_seconds * 1000.0 << "\n";
  std::cout << "# max_spread_pct=" << cfg.max_spread_pct << "\n";
  if (!cfg.csv_file.empty()) std::cout << "# csv=" << cfg.csv_file << "\n";
  std::cout << "# group=" << cfg.group << "\n";
}

// =========================================================================
// Main (outside anonymous namespace — must have external linkage)
// =========================================================================

template <typename Device>
int runSuite(const BenchConfig& cfg, const Device& device, const char* tensor_device) {
  if (cfg.verify) {
    std::vector<ProblemSize> verify_sizes;
    if (cfg.has_custom_size) {
      verify_sizes.push_back({"custom", cfg.custom_m, cfg.custom_k, cfg.custom_n});
    } else if (!cfg.csv_file.empty()) {
      if (!readCsv(cfg.csv_file, verify_sizes)) return 2;
    } else {
      for (int i = 0; i < numVerificationSizes(); ++i) {
        if (cfg.group == "all" || cfg.group == kVerificationSizes[i].group) {
          verify_sizes.push_back(kVerificationSizes[i]);
        }
      }
    }
    if (verify_sizes.empty()) {
      std::cerr << "No verification cases matched --group " << cfg.group << "\n";
      return 2;
    }

    std::cout << "# Verification mode: comparing Eigen contraction against double-precision GEMM\n";
    std::cout << "# compiler=" << __VERSION__ << "\n";
    std::cout << "# vector_backend=" << vectorBackend() << "\n";
    std::cout << "# vector_bits=" << vectorBits() << "\n";
    std::cout << "# kgemm_tensor_kernel=" << (kgemmTensorKernelEnabled() ? 1 : 0) << "\n";
    std::cout << "# kgemm_pack_reuse=" << (kgemmPackingReuseEnabled() ? 1 : 0) << "\n";
    std::cout << "# kgemm_test_instrumentation=" << (kgemmInstrumentationEnabled() ? 1 : 0) << "\n";
#if defined(EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN)
    std::cout << "# kgemm_pack_reuse_min_mn=" << EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN << "\n";
    std::cout << "# kgemm_pack_reuse_min_k=" << EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K << "\n";
#endif
    std::cout << "# tensor_device=" << tensor_device << "\n";
    std::cout << "# threads=" << cfg.threads << "\n";
    std::cout << "# layout=" << cfg.layout_str << "\n";
    std::cout << "# mode=" << (cfg.verify_both_modes ? "both" : (cfg.accumulate ? "cpeqab" : "ceqab")) << "\n";
    std::cout << "# tolerance=8*(gamma_(2K+1)*sum_abs+u*max(1,abs(reference)))\n";
    std::cout << "# input_pattern=deterministic_normal_v2\n";
    std::cout << "# exact_input_pattern=one_hot_a_integral_b_c\n";
    std::cout << "# initial_c=deterministic_nonzero\n";
    std::cout << "# tensor_scope=owning_contiguous_alpha1\n\n";

    int passed = 0;
    int failed = 0;
    for (const auto& sz : verify_sizes) {
      const int mode_count = cfg.verify_both_modes ? 2 : 1;
      for (int mode_index = 0; mode_index < mode_count; ++mode_index) {
        const bool accumulate = cfg.verify_both_modes ? mode_index == 1 : cfg.accumulate;
        if (cfg.layout_str == "row" || cfg.layout_str == "both") {
          const bool ok = verifyContraction<Eigen::RowMajor>(sz.m, sz.k, sz.n, accumulate, device);
          ok ? ++passed : ++failed;
        }
        if (cfg.layout_str == "col" || cfg.layout_str == "both") {
          const bool ok = verifyContraction<Eigen::ColMajor>(sz.m, sz.k, sz.n, accumulate, device);
          ok ? ++passed : ++failed;
        }
      }
    }
    std::cout << "\n# Result: " << passed << " passed, " << failed << " failed"
              << (failed == 0 ? "  ALL GOOD\n" : "  ERRORS DETECTED\n");
    return failed > 0 ? 1 : 0;
  }

  printMetadata(cfg, tensor_device);
  std::cout << "group,m,k,n,iterations,median_gflops,min_gflops,max_gflops,checksum,spread_pct,kernel_path\n";
  std::cout << std::fixed << std::setprecision(6);

  std::vector<ProblemSize> problems;
  if (cfg.has_custom_size) {
    problems.push_back({"custom", cfg.custom_m, cfg.custom_k, cfg.custom_n});
  } else if (!cfg.csv_file.empty()) {
    if (!readCsv(cfg.csv_file, problems)) return 2;
  } else {
    for (int i = 0; i < numBuiltinSizes(); ++i) problems.push_back(kBuiltinSizes[i]);
  }

  int problems_run = 0;
  for (const auto& sz : problems) {
    if (cfg.group == "all" || cfg.group == sz.group) {
      if (!runBenchmarkProblem(sz, cfg, device)) return 1;
      ++problems_run;
    }
  }
  if (problems_run == 0) {
    std::cerr << "No benchmark cases matched --group " << cfg.group << "\n";
    return 2;
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  BenchConfig cfg;
  if (!parseOptions(argc, argv, cfg)) {
    std::cerr << "Invalid arguments. Use --help for usage.\n";
    return 2;
  }

  // Validate kernel selection
  const char* backend = vectorBackend();
  if (!cfg.kernel_str.empty()) {
    bool ok = false;
    if (cfg.kernel_str == "neon") {
      ok = (std::strcmp(backend, "NEON") == 0) && !kgemmTensorKernelEnabled();
    } else if (cfg.kernel_str == "kgemm") {
      ok = kgemmTensorKernelEnabled();
    } else if (cfg.kernel_str == "sve") {
      ok = optimizedSve2GebpEnabled();
    }
    if (!ok) {
      std::cerr << "Error: --kernel " << cfg.kernel_str << " requested but binary compiled with " << backend
                << " backend.\n";
      std::cerr << "Recompile with appropriate flags:\n";
      std::cerr << "  NEON: -pthread -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64 -march=armv9-a+nosve\n";
      std::cerr << "  KGemm: -pthread -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64"
                   " -DEIGEN_NEON_USE_KGEMM=1 -march=armv9-a+nosve\n";
      std::cerr << "  SVE2: -pthread -DEIGEN_USE_THREADS=1 -DEIGEN_MAX_ALIGN_BYTES=64"
                   " -DEIGEN_ARM64_USE_SVE=1 -DEIGEN_BENCHMARK_REQUIRE_SVE2=1"
                   " -march=armv9-a+sve2 -msve-vector-bits=256\n";
      return 1;
    }
  }

  // Keep Eigen Core products serial inside TensorContraction worker tasks.
  // ThreadPoolDevice owns the only parallel scheduling layer when threads > 1.
  Eigen::setNbThreads(1);

  if (cfg.threads == 1) {
    Eigen::DefaultDevice device;
    return runSuite(cfg, device, "DefaultDevice");
  }

  Eigen::ThreadPool pool(cfg.threads);
  Eigen::ThreadPoolDevice device(&pool, cfg.threads);
  return runSuite(cfg, device, "ThreadPoolDevice");
}
