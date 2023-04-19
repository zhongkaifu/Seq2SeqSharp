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
    [CudaInclude("Code", "General")]
    public static class KernelGeneral
    {
        public static readonly string Code = @"

#define __int64 long long
#define __int32 int

#define MAX_CUTORCH_DIMS " + TSCudaContext.MaxDims + "\n" + @"

template <typename IndexType>
struct TensorInfo {
  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};
";

    }
}
