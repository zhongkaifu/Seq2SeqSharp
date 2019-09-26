#pragma once

#include <string.h>
#include <math.h>
#include <algorithm>

#include "General.h"
#include "TensorRef.h"
#include "Vector-inl.h"


OPS_API int TS_Unfolded_Copy(
	TensorRef* finput,
	TensorRef* input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight);

OPS_API int TS_Unfolded_Acc(
	TensorRef *finput,
	TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight);

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

OPS_API int TS_Softmax(
	TensorRef* out_,
	TensorRef* in_,
	int rows,
	int cols);

OPS_API int TS_SoftmaxGrad(
	TensorRef* grad_,
	TensorRef* adj_,
	TensorRef* val_,
	int rows,
	int cols,
	bool addGrad);

OPS_API int TS_RMSProp(
	TensorRef* tw,
	TensorRef* tg,
	TensorRef* tc,
	int rows,
	int cols,
	int batchSize,
	float step_size,
	float clipval,
	float regc,
	float decay_rate,
	float eps);


template<typename T>
void RMSProp(TensorRef* tw, TensorRef* tg, TensorRef* tc, int rows, int cols, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
{
	T * w = (T*)tw->buffer;
	T * g = (T*)tg->buffer;
	T * c = (T*)tc->buffer;

	for (int j = 0; j < rows; j++)
	{
		T * sw = w + j * cols;
		T * sg = g + j * cols;
		T * sc = c + j * cols;

		for (int i = 0; i < cols; i++)
		{
			if (sg[i] != 0.0)
			{
				T g = sg[i] / batchSize;

				if (g > clipval)
				{
					g = clipval;
				}
				if (g < -clipval)
				{
					g = -clipval;
				}

				sc[i] = sc[i] * decay_rate + (1.0 - decay_rate) * g * g;

				g = g / sqrt(sc[i] + eps);

				sw[i] -= g * step_size + sw[i] * regc;

				sg[i] = 0;
			}
		}
	}
}

template<typename T>
void Softmax(TensorRef* out, TensorRef* in, int rows, int cols) {
	T * pOut = (T*)out->buffer;
	T * pIn = (T*)in->buffer;

	for (int j = 0; j < rows; ++j) {
		T * so = pOut + j * cols;
		T * sp = pIn + j * cols;

		T max = sp[0];
		for (int i = 1; i < cols; ++i)
			max = std::max(max, sp[i]);

		T sum = 0.f;
		for (int i = 0; i < cols; ++i) {
			T ex = expf(sp[i] - max);
			so[i] = ex;
			sum += ex;
		}

		for (int i = 0; i < cols; ++i) {
			so[i] /= sum;
		}
	}
}

template<typename T>
void SoftmaxGrad(TensorRef* grad_, TensorRef* adj_, TensorRef* val_, int rows, int cols, bool addGrad) {

	T * grad = (T*)grad_->buffer;
	T * adj = (T*)adj_->buffer;
	T * val = (T*)val_->buffer;

	for (int j = 0; j < rows; ++j) {
		T * gradRow = grad + j * cols;
		T * adjRow = adj + j * cols;
		T * valRow = val + j * cols;

		T sum = 0.f;
		for (int i = 0; i < cols; ++i) {
			sum += valRow[i] * adjRow[i];
		}

		for (int i = 0; i < cols; ++i) {
			if (addGrad)
			{
				gradRow[i] += valRow[i] * (adjRow[i] - sum);
			}
			else
			{
				gradRow[i] = valRow[i] * (adjRow[i] - sum);
			}
		}
	}
}

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

// note: due to write issues, this one cannot be parallelized as well as unfolded_copy
template<typename T>
void unfolded_acc(
	TensorRef *finput,
	TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight)
{
	size_t nip;

	T *input_data = (T*)input->buffer;
	T *finput_data = (T*)finput->buffer;

#pragma omp parallel for private(nip)
	for (nip = 0; nip < nInputPlane; nip++)
	{
		size_t kw, kh, y, x;
		__int64 ix = 0, iy = 0;
		for (kh = 0; kh < kH; kh++)
		{
			for (kw = 0; kw < kW; kw++)
			{
				T *src = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
				T *dst = input_data + nip*(inputHeight*inputWidth);
				if (padW > 0 || padH > 0) {
					size_t lpad, rpad;
					for (y = 0; y < outputHeight; y++) {
						iy = (__int64)(y*dH - padH + kh);
						if (iy < 0 || iy >= inputHeight) {
						}
						else {
							if (dW == 1) {
								ix = (__int64)(0 - padW + kw);
								lpad = std::max(size_t(0), (padW - kw));
								rpad = std::max(size_t(0), (padW - (kW - kw - 1)));
								Vector_add<T>(dst + (size_t)(iy*inputWidth + ix + lpad), src + (size_t)(y*outputWidth + lpad), 1, outputWidth - lpad - rpad);
							}
							else {
								for (x = 0; x<outputWidth; x++) {
									ix = (__int64)(x*dW - padW + kw);
									if (ix < 0 || ix >= inputWidth) {
									}
									else
										Vector_add<T>(dst + (size_t)(iy*inputWidth + ix), src + (size_t)(y*outputWidth + x), 1, 1);
								}
							}
						}
					}
				}
				else {
					for (y = 0; y < outputHeight; y++) {
						iy = (__int64)(y*dH + kh);
						ix = (__int64)(0 + kw);
						if (dW == 1)
							Vector_add<T>(dst + (size_t)(iy*inputWidth + ix), src + (size_t)(y*outputWidth), 1, outputWidth);
						else {
							for (x = 0; x < outputWidth; x++)
								Vector_add<T>(dst + (size_t)(iy*inputWidth + ix + x*dW), src + (size_t)(y*outputWidth + x), 1, 1);
						}
					}
				}
			}
		}
	}
}



template<typename T>
void unfolded_copy(TensorRef *finput, TensorRef *input,
	int kW,
	int kH,
	int dW,
	int dH,
	int padW,
	int padH,
	int nInputPlane,
	int inputWidth,
	int inputHeight,
	int outputWidth,
	int outputHeight)
{
	long k;
	T *input_data = (T*)input->buffer;
	T *finput_data = (T*)finput->buffer;

#pragma omp parallel for private(k)
	for (k = 0; k < nInputPlane*kH*kW; k++) {
		size_t nip = k / (kH*kW);
		size_t rest = k % (kH*kW);
		size_t kh = rest / kW;
		size_t kw = rest % kW;
		size_t x, y;
		__int64 ix, iy;
		T *dst = finput_data + nip*(kH*kW*outputHeight*outputWidth) + kh*(kW*outputHeight*outputWidth) + kw*(outputHeight*outputWidth);
		T *src = input_data + nip*(inputHeight*inputWidth);
		if (padW > 0 || padH > 0) {
			size_t lpad, rpad;
			for (y = 0; y < outputHeight; y++) {
				iy = (__int64)(y*dH - padH + kh);
				if (iy < 0 || iy >= inputHeight) {
					memset(dst + y*outputWidth, 0, sizeof(T)*outputWidth);
				}
				else {
					if (dW == 1) {
						ix = (__int64)(0 - padW + kw);
						lpad = std::max(size_t(0), (padW - kw));
						rpad = std::max(size_t(0), (padW - (kW - kw - 1)));
						if (outputWidth - rpad - lpad <= 0) {
							memset(dst + (size_t)(y*outputWidth), 0, sizeof(T)*outputWidth);
						}
						else {
							if (lpad > 0) memset(dst + y*outputWidth, 0, sizeof(T)*lpad);
							memcpy(dst + (size_t)(y*outputWidth + lpad), src + (size_t)(iy*inputWidth + ix + lpad), sizeof(T)*(outputWidth - rpad - lpad));
							if (rpad > 0) memset(dst + y*outputWidth + outputWidth - rpad, 0, sizeof(T)*rpad);
						}
					}
					else {
						for (x = 0; x<outputWidth; x++) {
							ix = (__int64)(x*dW - padW + kw);
							if (ix < 0 || ix >= inputWidth)
								memset(dst + (size_t)(y*outputWidth + x), 0, sizeof(T) * 1);
							else
								memcpy(dst + (size_t)(y*outputWidth + x), src + (size_t)(iy*inputWidth + ix), sizeof(T)*(1));
						}
					}
				}
			}
		}
		else {
			for (y = 0; y < outputHeight; y++) {
				iy = (__int64)(y*dH + kh);
				ix = (__int64)(0 + kw);
				if (dW == 1)
					memcpy(dst + (size_t)(y*outputWidth), src + (size_t)(iy*inputWidth + ix), sizeof(T)*outputWidth);
				else {
					for (x = 0; x<outputWidth; x++)
						memcpy(dst + (size_t)(y*outputWidth + x), src + (size_t)(iy*inputWidth + ix + x*dW), sizeof(T)*(1));
				}
			}
		}
	}
}