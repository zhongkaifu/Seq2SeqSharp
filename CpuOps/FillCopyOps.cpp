//// CpuOps.cpp : Defines the exported functions for the DLL application.
////
//
//#include "FillCopyOps.h"
//#include "TensorIter-inl.h"
//#include "TensorApply-inl.h"
//
//
//
//template<typename TResult, typename TSrc>
//INLINE_FUNC void Copy_Apply(TensorRef* result, TensorRef* src)
//{
//	auto func = [](TResult *r, TSrc *s) { *r = (TResult)*s; };
//	Apply2<TResult, TSrc>(result, src, func);
//}
//
//
//template<typename TResult>
//INLINE_FUNC void Copy_ToResult(TensorRef* result, TensorRef* src)
//{
//	switch (src->elementType)
//	{
//	case DType::Float32: Copy_Apply<TResult, float>(result, src); break;
//	case DType::Float64: Copy_Apply<TResult, double>(result, src); break;
//	case DType::Int32: Copy_Apply<TResult, __int32>(result, src); break;
//	case DType::UInt8: Copy_Apply<TResult, uint8>(result, src); break;
//	default:
//		throw TSError("Tensor type not supported for Copy");\
//		break;
//	}
//}
//
//int TS_Copy(TensorRef* result, TensorRef* src)
//{
//	API_BEGIN()
//	switch (result->elementType)
//	{
//	case DType::Float32: Copy_ToResult<float>(result, src); break;
//	case DType::Float64: Copy_ToResult<double>(result, src); break;
//	case DType::Int32: Copy_ToResult<__int32>(result, src); break;
//	case DType::UInt8: Copy_ToResult<uint8>(result, src); break;
//	default:
//		throw TSError("Tensor type not supported for Copy");\
//		break;
//	}
//	API_END()
//}
//
