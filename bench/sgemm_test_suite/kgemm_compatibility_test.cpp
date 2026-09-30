// SPDX-License-Identifier: MPL-2.0
// Regression tests for the TensorFlow 2.20 Eigen baseline. C++14, -pthread, -I.
// ARM verification: add -DEIGEN_NEON_USE_KGEMM=1.
// To exercise packed scheduling with small inputs also add:
// -DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN=32 -DEIGEN_NEON_KGEMM_PACK_REUSE_MIN_K=32
// -DEIGEN_NEON_KGEMM_PACK_KC=17
// This instrumented executable is NOT a performance benchmark.
#define EIGEN_USE_THREADS 1
#define EIGEN_NEON_KGEMM_TEST_INSTRUMENTATION 1
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>

#if EIGEN_ARCH_ARM64 && defined(EIGEN_VECTORIZE_NEON) && defined(EIGEN_NEON_USE_KGEMM) && EIGEN_NEON_USE_KGEMM
#define TEST_KGEMM_ACTIVE 1
#else
#define TEST_KGEMM_ACTIVE 0
#endif

struct ExternalKernelContract { enum { HasBeta = true }; };
struct KGemmKernelContract { enum { HasBeta = true, IsKGemm = true }; };
static_assert(!Eigen::internal::TensorContractionKernelIsKGemm<ExternalKernelContract>::value,
              "External kernels need not implement IsKGemm");
static_assert(Eigen::internal::TensorContractionKernelIsKGemm<KGemmKernelContract>::value,
              "KGEMM must select its own blocking");

template <typename Output, typename Expression>
void assign(Output& out, const Expression& expression, const Eigen::DefaultDevice& device, bool) {
  out.device(device) = expression;
}

template <typename Output, typename Expression>
void assign(Output& out, const Expression& expression, const Eigen::ThreadPoolDevice& device, bool async) {
  if (async) {
    Eigen::Barrier done(1);
    out.device(device, [&done]() { done.Notify(); }) = expression;
    done.Wait();
  } else {
    out.device(device) = expression;
  }
}

template <typename Scalar, int Layout, typename Device>
bool check(const Device& device, bool async, int m, int k, int n, bool ta, bool tb, bool expression) {
  typedef Eigen::Tensor<Scalar, 2, Layout> Tensor;
  Tensor a(ta ? k : m, ta ? m : k), b(tb ? n : k, tb ? k : n), c(m, n);
  for (int i = 0; i < m; ++i)
    for (int p = 0; p < k; ++p) {
      const Scalar value = Scalar((i * 3 + p * 5) % 17 - 8) / Scalar(16);
      if (ta) a(p, i) = value; else a(i, p) = value;
    }
  for (int p = 0; p < k; ++p)
    for (int j = 0; j < n; ++j) {
      const Scalar value = Scalar((p * 7 + j * 3) % 19 - 9) / Scalar(16);
      if (tb) b(j, p) = value; else b(p, j) = value;
    }
  Eigen::array<Eigen::IndexPair<int>, 1> dims = {Eigen::IndexPair<int>(ta ? 0 : 1, tb ? 1 : 0)};
#if TEST_KGEMM_ACTIVE
  Eigen::internal::kgemmTestResetInvocationCounters();
#endif
  // Overwrite nonzero output twice to catch stale C / unintended accumulation.
  for (int repeat = 0; repeat < 2; ++repeat) {
    c.setConstant(Scalar(7));
    if (expression) assign(c, (a + a).contract(b, dims), device, async);
    else assign(c, a.contract(b, dims), device, async);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double expected = 0;
        for (int p = 0; p < k; ++p)
          expected += double(ta ? a(p, i) : a(i, p)) * double(tb ? b(j, p) : b(p, j));
        if (expression) expected *= 2;
        // These small dyadic inputs/sums are exactly representable in FP32.
        if (!std::isfinite(double(c(i, j))) || double(c(i, j)) != expected) {
          std::cerr << "FAIL value m,k,n=" << m << ',' << k << ',' << n << " ta,tb=" << ta << ',' << tb
                    << " layout=" << Layout << " async=" << async << " expression=" << expression << '\n';
          return false;
        }
      }
  }
#if TEST_KGEMM_ACTIVE
  const auto raw = Eigen::internal::kgemmTestRawInvocationCount();
  const auto packed = Eigen::internal::kgemmTestPackedInvocationCount();
  const bool eligible = std::is_same<Scalar, float>::value && !ta && !tb && !expression && m > 0 && k > 0 && n > 0 &&
                        (Layout == Eigen::RowMajor ? m != 1 : n != 1);
  bool expect_packed = false;
#if EIGEN_NEON_KGEMM_REUSE_PACKING
  expect_packed = eligible && std::min(m, n) >= EIGEN_NEON_KGEMM_PACK_REUSE_MIN_MN &&
                  k >= EIGEN_NEON_KGEMM_PACK_REUSE_MIN_K;
#endif
  if ((!eligible && raw + packed != 0) || (eligible && expect_packed && (packed == 0 || raw != 0)) ||
      (eligible && !expect_packed && (raw == 0 || packed != 0))) {
    std::cerr << "FAIL dispatch m,k,n=" << m << ',' << k << ',' << n << " raw=" << raw << " packed=" << packed
              << " eligible=" << eligible << " expect_packed=" << expect_packed << '\n';
    return false;
  }
#endif
  return true;
}

template <int Layout, typename Device>
bool suite(const Device& device, bool async) {
  const int shapes[][3] = {{1, 24, 16}, {35, 37, 1}, {23, 37, 29}, {145, 129, 161}, {65, 513, 67},
                          {16, 0, 16}, {0, 24, 16}, {16, 24, 0}};
  for (const auto& s : shapes)
    if (!check<float, Layout>(device, async, s[0], s[1], s[2], false, false, false)) return false;
  for (int ta = 0; ta < 2; ++ta)
    for (int tb = 0; tb < 2; ++tb)
      if (!check<float, Layout>(device, async, 67, 131, 71, ta, tb, false)) return false;
  return check<float, Layout>(device, async, 67, 131, 71, false, false, true) &&
         check<double, Layout>(device, async, 67, 131, 71, false, false, false);
}

int main() {
  bool ok = suite<Eigen::RowMajor>(Eigen::DefaultDevice(), false) &&
            suite<Eigen::ColMajor>(Eigen::DefaultDevice(), false);
  for (int threads : {1, 2, 4, 8}) {
    Eigen::ThreadPool pool(threads);
    Eigen::ThreadPoolDevice device(&pool, threads);
    for (bool async : {false, true})
      ok = suite<Eigen::RowMajor>(device, async) && suite<Eigen::ColMajor>(device, async) && ok;
  }
  std::cout << (ok ? "PASS" : "FAIL") << ": DefaultDevice + ThreadPool(1/2/4/8), sync/async, row/col, NN/NT/TN/TT, "
            << "expression, FP64, zero dimensions; kgemm_active=" << TEST_KGEMM_ACTIVE << '\n';
  return ok ? 0 : 1;
}
