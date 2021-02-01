
#include "ConvolutionUnfold.h"


int TS_Unfolded_Copy(TensorRef* finput, TensorRef* input,
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
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(input->elementType, unfolded_copy, finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
	API_END()
}

int TS_Unfolded_Acc(
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
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(input->elementType, unfolded_acc, finput, input, kW, kH, dW, dH, padW, padH, nInputPlane, inputWidth, inputHeight, outputWidth, outputHeight)
	API_END()
}

int TS_Softmax(
	TensorRef* out_,
	TensorRef* in_,
	int rows,
	int cols)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(in_->elementType, Softmax, out_, in_, rows, cols)
		API_END()
}


int TS_SoftmaxGrad(
	TensorRef* grad_, 
	TensorRef* adj_, 
	TensorRef* val_, 
	int rows, 
	int cols,
	bool addGrad)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(grad_->elementType, SoftmaxGrad, grad_, adj_, val_, rows, cols, addGrad)
		API_END()
}