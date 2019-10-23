#pragma once

#include <string.h>
#include <math.h>
#include <algorithm>

#include "General.h"
#include "TensorRef.h"
#include "Vector-inl.h"

OPS_API int TS_LayerNorm(
	TensorRef* out_,
	TensorRef* in_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols);

OPS_API int TS_LayerNormGrad(
	TensorRef * gradX_,
	TensorRef * gradGamma_,
	TensorRef * gradBeta_,
	TensorRef * adj_,
	TensorRef * y_,
	TensorRef * x_,
	TensorRef * gamma_,
	TensorRef * beta_,
	int rows,
	int cols,
	float eps);

OPS_API int TS_AddLayerNorm(
	TensorRef* out_,
	TensorRef* in1_,
	TensorRef* in2_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols);

OPS_API int TS_AddLayerNormGrad(
	TensorRef * gradX1_,
	TensorRef * gradX2_,
	TensorRef * gradGamma_,
	TensorRef * gradBeta_,
	TensorRef * adj_,
	TensorRef * y_,
	TensorRef * x1_,
	TensorRef * x2_,
	TensorRef * gamma_,
	TensorRef * beta_,
	int rows,
	int cols,
	float eps);

template<typename T>
void LayerNorm(TensorRef* out_,
	TensorRef* in_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols) {
	T * out = (T*)out_->buffer;
	T * in = (T*)in_->buffer;
	T * alpha = (T*)gamma_->buffer;
	T * beta = beta_ ? (T*)beta_->buffer : nullptr;

#pragma omp parallel for
	for (int j = 0; j < rows; ++j) {
		T * so = out + j * cols;
		const T * sp = in + j * cols;

		T sum = 0.f;
#pragma omp simd reduction(+ : sum)
		for (int i = 0; i < cols; ++i) {
			sum += sp[i];
		}

		T mean = sum / cols;
		T sqSum = 0.f;
#pragma omp simd reduction(+ : sqSum)
		for (int i = 0; i < cols; ++i) {
			T ex = sp[i] - mean;
			sqSum += ex * ex;
		}

		T sigma = std::sqrt(eps + sqSum / cols);

#pragma omp simd
		for (int i = 0; i < cols; ++i) {
			T t = alpha[i] * ((sp[i] - mean) / sigma);
			if (beta != nullptr) {
				t += beta[i];
			}

			so[i] = t;
		}
	}
}

template<typename T>
void LayerNormGrad(TensorRef * gradX_,
	TensorRef * gradGamma_,
	TensorRef * gradBeta_,
	TensorRef * adj_,
	TensorRef * y_,
	TensorRef * x_,
	TensorRef * gamma_,
	TensorRef * beta_,
	int rows,
	int cols,
	float eps) {
	T * gradX = (T*)gradX_->buffer;
	T * gradGamma = (T*)gradGamma_->buffer;
	T * gradBeta = gradBeta_ ? (T*)gradBeta_->buffer : nullptr;
	T * adj = (T*)adj_->buffer;
	T * y = (T*)y_->buffer;
	T * x = (T*)x_->buffer;
	T * gamma = (T*)gamma_->buffer;
	T * beta = beta_ ? (T*)beta_->buffer : nullptr;

	if (beta) {
#pragma omp parallel for reduction(+ : gradGamma[:cols], gradBeta[:cols])
		for (size_t j = 0; j < rows; ++j) {
			T * xRow = x + j * cols;
			T * yRow = y + j * cols;
			T * adjRow = adj + j * cols;
			T * gradXRow = gradX + j * cols;

			T sum_x = 0.f;
			T sum_adj = 0.f;
			T sum_adj_x = 0.f;
			T sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
			for (size_t i = 0; i < cols; ++i) {
				sum_x += xRow[i];
				sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
				sum_adj += adjRow[i];
			}

			T mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
			for (size_t i = 0; i < cols; ++i) {
				T ex = xRow[i] - mean;
				sum_sqr += ex * ex;
			}

			T sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
			for (size_t i = 0; i < cols; ++i) {
				T grad_x = 0.f;
				T x_hat = (yRow[i] - beta[i]) / gamma[i];
				grad_x += cols * adjRow[i];
				grad_x -= sum_adj;
				grad_x -= sum_adj_x * x_hat;
				grad_x /= cols * sigma;

				gradXRow[i] += gamma[i] * grad_x;
				gradGamma[i] += adjRow[i] * x_hat;
				gradBeta[i] += adjRow[i];
			}
		}
	}
	else {
#pragma omp parallel for reduction(+ : gradGamma[:cols])
		for (size_t j = 0; j < rows; ++j) {
			T * xRow = x + j * cols;
			T * yRow = y + j * cols;
			T * adjRow = adj + j * cols;
			T *gradXRow = gradX + j * cols;

			T sum_x = 0.f;
			T sum_adj = 0.f;
			T sum_adj_x = 0.f;
			T sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
			for (size_t i = 0; i < cols; ++i) {
				sum_x += xRow[i];
				sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
				sum_adj += adjRow[i];
			}

			T mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
			for (size_t i = 0; i < cols; ++i) {
				T ex = xRow[i] - mean;
				sum_sqr += ex * ex;
			}

			T sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
			for (size_t i = 0; i < cols; ++i) {
				T grad_x = 0.f;
				T x_hat = yRow[i] / gamma[i];
				grad_x += cols * adjRow[i];
				grad_x -= sum_adj;
				grad_x -= sum_adj_x * x_hat;
				grad_x /= cols * sigma;

				gradXRow[i] += gamma[i] * grad_x;
				gradGamma[i] += adjRow[i] * x_hat;
			}
		}
	}
}


template<typename T>
void AddLayerNorm(TensorRef* out_,
	TensorRef* in1_,
	TensorRef* in2_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols) {
	T * out = (T*)out_->buffer;
	T * in1 = (T*)in1_->buffer;
	T * in2 = (T*)in2_->buffer;
	T * alpha = (T*)gamma_->buffer;
	T * beta = beta_ ? (T*)beta_->buffer : nullptr;

#pragma omp parallel for
	for (int j = 0; j < rows; ++j) {
		T * so = out + j * cols;
		const T * sp1 = in1 + j * cols;
		const T * sp2 = in2 + j * cols;

		T sum = 0.f;
#pragma omp simd reduction(+ : sum)
		for (int i = 0; i < cols; ++i) {
			sum += (sp1[i] + sp2[i]);
		}

		T mean = sum / cols;
		T sqSum = 0.f;
#pragma omp simd reduction(+ : sqSum)
		for (int i = 0; i < cols; ++i) {
			T ex = (sp1[i] + sp2[i]) - mean;
			sqSum += ex * ex;
		}

		T sigma = std::sqrt(eps + sqSum / cols);

#pragma omp simd
		for (int i = 0; i < cols; ++i) {
			T t = alpha[i] * (((sp1[i] + sp2[i]) - mean) / sigma);
			if (beta != nullptr) {
				t += beta[i];
			}

			so[i] = t;
		}
	}
}

template<typename T>
void AddLayerNormGrad(
	TensorRef * gradX1_,
	TensorRef * gradX2_,
	TensorRef * gradGamma_,
	TensorRef * gradBeta_,
	TensorRef * adj_,
	TensorRef * y_,
	TensorRef * x1_,
	TensorRef * x2_,
	TensorRef * gamma_,
	TensorRef * beta_,
	int rows,
	int cols,
	float eps) {
	T * gradX1 = (T*)gradX1_->buffer;
	T * gradX2 = (T*)gradX2_->buffer;
	T * gradGamma = (T*)gradGamma_->buffer;
	T * gradBeta = gradBeta_ ? (T*)gradBeta_->buffer : nullptr;
	T * adj = (T*)adj_->buffer;
	T * y = (T*)y_->buffer;
	T * x1 = (T*)x1_->buffer;
	T * x2 = (T*)x2_->buffer;
	T * gamma = (T*)gamma_->buffer;
	T * beta = beta_ ? (T*)beta_->buffer : nullptr;

	if (beta) {
#pragma omp parallel for reduction(+ : gradGamma[:cols], gradBeta[:cols])
		for (size_t j = 0; j < rows; ++j) {
			T * x1Row = x1 + j * cols;
			T * x2Row = x2 + j * cols;
			T * yRow = y + j * cols;
			T * adjRow = adj + j * cols;
			T * gradX1Row = gradX1 + j * cols;
			T * gradX2Row = gradX2 + j * cols;

			T sum_x = 0.f;
			T sum_adj = 0.f;
			T sum_adj_x = 0.f;
			T sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
			for (size_t i = 0; i < cols; ++i) {
				sum_x += (x1Row[i] + x2Row[i]);
				sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
				sum_adj += adjRow[i];
			}

			T mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
			for (size_t i = 0; i < cols; ++i) {
				T ex = (x1Row[i] + x2Row[i]) - mean;
				sum_sqr += ex * ex;
			}

			T sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
			for (size_t i = 0; i < cols; ++i) {
				T grad_x = 0.f;
				T x_hat = (yRow[i] - beta[i]) / gamma[i];
				grad_x += cols * adjRow[i];
				grad_x -= sum_adj;
				grad_x -= sum_adj_x * x_hat;
				grad_x /= cols * sigma;

				gradX1Row[i] += gamma[i] * grad_x;
				gradX2Row[i] += gamma[i] * grad_x;
				gradGamma[i] += adjRow[i] * x_hat;
				gradBeta[i] += adjRow[i];
			}
		}
	}
	else {
#pragma omp parallel for reduction(+ : gradGamma[:cols])
		for (size_t j = 0; j < rows; ++j) {
			T * x1Row = x1 + j * cols;
			T * x2Row = x2 + j * cols;
			T * yRow = y + j * cols;
			T * adjRow = adj + j * cols;
			T *gradX1Row = gradX1 + j * cols;
			T *gradX2Row = gradX2 + j * cols;

			T sum_x = 0.f;
			T sum_adj = 0.f;
			T sum_adj_x = 0.f;
			T sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
			for (size_t i = 0; i < cols; ++i) {
				sum_x += (x1Row[i] + x2Row[i]);
				sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
				sum_adj += adjRow[i];
			}

			T mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
			for (size_t i = 0; i < cols; ++i) {
				T ex = (x1Row[i] + x2Row[i]) - mean;
				sum_sqr += ex * ex;
			}

			T sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
			for (size_t i = 0; i < cols; ++i) {
				T grad_x = 0.f;
				T x_hat = yRow[i] / gamma[i];
				grad_x += cols * adjRow[i];
				grad_x -= sum_adj;
				grad_x -= sum_adj_x * x_hat;
				grad_x /= cols * sigma;

				gradX1Row[i] += gamma[i] * grad_x;
				gradX2Row[i] += gamma[i] * grad_x;
				gradGamma[i] += adjRow[i] * x_hat;
			}
		}
	}
}