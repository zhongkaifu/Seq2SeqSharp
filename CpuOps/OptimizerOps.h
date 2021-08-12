#pragma once

#include <string.h>
#include <math.h>
#include <algorithm>

#include "General.h"
#include "TensorRef.h"
#include "Vector-inl.h"

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

//OPS_API int TS_Adam(
//	TensorRef* tw,
//	TensorRef* tg,
//	TensorRef* tv,
//	TensorRef* tm,
//	int rows,
//	int cols,
//	int batchSize,
//	float step_size,
//	float clipval,
//	float regc,
//	float decay_rate_v,
//	float decay_rate_m,
//	int iter,
//	float eps);
//
//
//template<typename T>
//void Adam(TensorRef* tw, TensorRef* tg, TensorRef* tv, TensorRef* tm, int rows, int cols, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
//{
//	T * w = (T*)tw->buffer;
//	T * g = (T*)tg->buffer;
//	T * v = (T*)tv->buffer;
//	T * m = (T*)tm->buffer;
//
//	for (int j = 0; j < rows; j++)
//	{
//		T * sw = w + j * cols;
//		T * sg = g + j * cols;
//		T * sv = v + j * cols;
//		T * sm = m + j * cols;
//
//		for (int i = 0; i < cols; i++)
//		{
//			if (sg[i] != 0.0)
//			{
//				T g = sg[i] / batchSize;
//
//				if (g > clipval)
//				{
//					g = clipval;
//				}
//				if (g < -clipval)
//				{
//					g = -clipval;
//				}
//
//				sm[i] = sm[i] * decay_rate_m + (1.0 - decay_rate_m) * g;
//				sv[i] = sv[i] * decay_rate_v + (1.0 - decay_rate_v) * g * g;
//
//				T m_cap = sm[i] / (1.0 - pow(decay_rate_m, iter));
//				T v_cap = sv[i] / (1.0 - pow(decay_rate_v, iter));
//
//				sw[i] -= step_size * m_cap / (sqrt(v_cap) + eps);
//
//				sg[i] = 0;
//			}
//		}
//	}
//}


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