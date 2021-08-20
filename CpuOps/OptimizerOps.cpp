#include "OptimizerOps.h"

int TS_RMSProp(
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
	float eps)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(tw->elementType, RMSProp, tw, tg, tc, rows, cols, batchSize, step_size, clipval, regc, decay_rate, eps)
		API_END()
}