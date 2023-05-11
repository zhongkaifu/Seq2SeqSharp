// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "Fp16")]
    public static class Fp16
    {
        public static readonly string Code = (TSCudaContext.ElementType == DType.Float16) ? "#include <cuda_fp16.h>" : "";
//typedef struct __align__(2) {
//   unsigned short x;
//} __half;
//typedef __half half;
//#define FP16_FUNC static __device__ __inline__
//FP16_FUNC __half __float2half(const float a);
//FP16_FUNC float __half2float(const __half a);
//
//";

    }
}
