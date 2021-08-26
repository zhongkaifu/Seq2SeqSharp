//#include "GatherScatterOps.h"
//#include "TensorIter-inl.h"
//#include "TensorApplyDim-inl.h"
//
//template<typename T>
//INLINE_FUNC void ScatterFill_Apply(TensorRef* result, float value, int dim, TensorRef* indices)
//{
//	auto func = [value](
//		T *rData, __int64 rSize, __int64 rStride,
//		T *iData, __int64 iSize, __int64 iStride)
//	{
//		for (int i = 0; i < iSize; ++i)
//		{
//			long idx = (long)*(iData + i * iStride);
//			if (idx < 0 || idx >= rSize) { throw TSError("Invalid index in gather"); }
//
//			rData[idx*rStride] = T(value);
//		}
//	};
//
//	ApplyDim2<T, T>(result, indices, dim, func);
//}
//
//int TS_ScatterFill(TensorRef* result, float value, int dim, TensorRef* indices)
//{
//	API_BEGIN()
//		SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, ScatterFill_Apply, result, value, dim, indices)
//		API_END()
//}
//
