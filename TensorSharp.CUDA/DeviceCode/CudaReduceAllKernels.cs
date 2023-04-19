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
    public class CudaReduceAllKernels : CudaCode
    {
        public CudaReduceAllKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "ReduceAll", "ReduceAllMacros", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            string identity = "return a;";

            PermutationGenerator result = new PermutationGenerator();
            result.AddReduceAll("sumAll", identity, "return a + b;");
            result.AddReduceAll("prodAll", identity, "return a * b;");
            result.AddReduceAll("minAll", identity, "return min(a, b);");
            result.AddReduceAll("maxAll", identity, "return max(a, b);");

            result.AddReduceAll("e0_normAll", "return a != 0 ? 1 : 0;", "return a + b;");
            result.AddReduceAll("e1_normAll", "return fabsf(a);", "return a + b;");
            result.AddReduceAll("e2_normAll", "return a * a;", "return a + b;");
            result.AddReduceAllNorm("en_normAll");

            result.AddReduceAllSubSquare("subSquare");

            return result.ToString();
        }
    }
}
