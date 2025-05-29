#include <armadillo>
#include <print>
#include <cmath>
#include <complex>

using cmat = arma::cx_mat; 
using cvec = arma::cx_vec;
using mat = arma::mat; 
using vec = arma::vec;

[[gnu::const]] arma::cx_double WNk(const double k, const double N) {
  double angle = -2.0 * arma::datum::pi * k / N;
  return std::polar(1.0, angle);
}

[[gnu::const]] arma::cx_double WNk_i(const double k, const double N) {
  double angle = 2.0 * arma::datum::pi * k/ N;
  return std::polar(1.0, angle);
}

template <bool inverse> 
auto fft1_impl(const cvec &A) {
  if (A.size() == 1) [[unlikely]]
    return A;

  const auto N = A.size();
  auto X = cvec(N);

  const arma::uvec even_indices =
    arma::regspace<arma::uvec>(0, 2, N - 1);

  const arma::uvec odd_indices = 
    arma::regspace<arma::uvec>(1, 2, N - 1);

  auto even = fft1_impl<inverse>(cvec(A(even_indices)));
  auto odd = fft1_impl<inverse>(cvec(A(odd_indices)));

  constexpr auto WFN = [] consteval -> auto {
    if constexpr (!inverse)
      return WNk;
    else
      return WNk_i;
  }();

  for (auto k = 0; k < N/2; k++) {
    auto W = WFN(k, N);
    X[k] = even[k] + (W * odd[k]);
    X[k + N/2] = even[k] - (W * odd[k]);
  }
  return X;
}

template <bool inverse = false> auto fft1(const cvec &x) {
  auto result = fft1_impl<inverse>(x);

  if constexpr (inverse)
    result /= x.size();

  return result;
}

template <bool inverse = false> auto fft1(const vec &x) {
  arma::cx_vec cx = arma::conv_to<arma::cx_vec>::from(x);
  return fft1<inverse>(cx);
}

int main(){
  vec x = arma::linspace(0.0, 100.0, std::pow(2,16));
  cvec fx = fft1(x);
  cvec ifx = fft1<true>(fx);

  x.print("OLD");
  ifx.print("REAL");

  return 0;
}
