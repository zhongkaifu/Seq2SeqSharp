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
    public class ElementwiseTriKernels : CudaCode
    {
        public ElementwiseTriKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            PermutationGenerator result = new PermutationGenerator();

            AppendTTFunc(result, "sin", "sin");
            AppendTTFunc(result, "cos", "cos");
            AppendTTFunc(result, "tan", "tan");
            AppendTTFunc(result, "asin", "asin");
            AppendTTFunc(result, "acos", "acos");
            AppendTTFunc(result, "atan", "atan");
            AppendTTFunc(result, "sinh", "sinh");
            AppendTTFunc(result, "cosh", "cosh");
            AppendTTFunc(result, "tanh", "tanhf");

            result.AddApplyTTT("atan2", "*a = atan2f(*b, *c);");

            AppendTTTFunc(result, "addtanh", "AddTanh");
            AppendTTTTFunc(result, "addtanh3", "AddTanh3");
            AppendTTTTFunc(result, "addtanhD", "AddTanhD");
            AppendTTTFunc(result, "tanhD", "TanhD");

            return result.ToString();
        }

        private static void AppendTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyT("t1_" + kernelBaseName, string.Format("*v = {0}(*v);", func));
            pg.AddApplyTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b);", func));
        }

        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b);", func));
            pg.AddApplyTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c);", func));
        }

        private static void AppendTTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, *c);", func));
            pg.AddApplyTTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, *d);", func));
        }

    }
}
