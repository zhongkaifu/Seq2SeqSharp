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