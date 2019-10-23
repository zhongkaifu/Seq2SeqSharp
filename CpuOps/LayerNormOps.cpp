#include "LayerNormOps.h"

int TS_LayerNorm(
	TensorRef* out_,
	TensorRef* in_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(in_->elementType, LayerNorm, out_, in_, gamma_, beta_, eps, rows, cols)
		API_END()
}


int TS_LayerNormGrad(
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
	float eps)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(gradX_->elementType, LayerNormGrad, gradX_, gradGamma_, gradBeta_, adj_, y_, x_, gamma_, beta_, rows, cols, eps)
		API_END()
}


int TS_AddLayerNorm(
	TensorRef* out_,
	TensorRef* in1_,
	TensorRef* in2_,
	TensorRef* gamma_,
	TensorRef* beta_,
	float eps,
	int rows,
	int cols)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(in1_->elementType, AddLayerNorm, out_, in1_, in2_, gamma_, beta_, eps, rows, cols)
		API_END()
}

int TS_AddLayerNormGrad(
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
	float eps)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(gradX1_->elementType, AddLayerNormGrad, gradX1_, gradX2_, gradGamma_, gradBeta_, adj_, y_, x1_, x2_, gamma_, beta_, rows, cols, eps)
		API_END()
}