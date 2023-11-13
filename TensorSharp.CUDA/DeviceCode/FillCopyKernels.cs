// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class FillCopyKernels : CudaCode
    {
        public FillCopyKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            PermutationGenerator result = new PermutationGenerator();
            result.AddApplyTS("fill", "*a = b;");
            result.AddApplyTT("copy", "*a = *b;");

            if (TSCudaContext.ElementType == DType.Float16)
            {
                Logger.WriteLine(Logger.Level.debug, $"Creating FillCopy kernels for Float16 type.");

                result.AddApplyTSHalf("fill", "*a = b;");
                result.AddApplyTT("copy", "*a = *b;", elementTypes: new DType[] { DType.Float16, DType.Float16 });
            }

            return result.ToString();
        }
    }

}
