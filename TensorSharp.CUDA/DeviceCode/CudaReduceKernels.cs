// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class CudaReduceKernels : CudaCode
    {
        public CudaReduceKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            string identity = "return a;";

            PermutationGenerator result = new PermutationGenerator();
            result.AddReduce("sum", identity, "return a + b;");
            result.AddReduce("prod", identity, "return a * b;");
            result.AddReduce("min", identity, "return min(a, b);");
            result.AddReduce("max", identity, "return max(a, b);");

            result.AddReduce("e0_norm", "return a != 0 ? 1 : 0;", "return a + b;");
            result.AddReduce("e1_norm", "return fabsf(a);", "return a + b;");
            result.AddReduce("e2_norm", "return a * a;", "return a + b;");
            result.AddReduceNorm("en_norm");

            return result.ToString();
        }
    }

}
