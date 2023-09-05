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
    public class ElementwiseActKernels : CudaCode
    {
        public ElementwiseActKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            PermutationGenerator result = new PermutationGenerator();

            AppendTTFunc(result, "sigmoid", "Sigmoid");
            AppendTTTTFunc(result, "addsigmoidD", "AddSigmoidD");
            AppendTTTFunc(result, "sigmoidD", "SigmoidD");

            AppendTTFunc(result, "relu", "relu");
            AppendTTTFunc(result, "relud", "relud");
            AppendTTTTFunc(result, "addrelud", "addrelud");

            AppendTTFunc(result, "SiLU", "SiLU");
            AppendTTTFunc(result, "SiLUD", "SiLUD");
            AppendTTTTFunc(result, "AddSiLUD", "AddSiLUD");


            AppendTTFunc(result, "LeakyReLU", "LeakyReLU");
            AppendTTTFunc(result, "LeakyReLUD", "LeakyReLUD");
            AppendTTTTFunc(result, "AddLeakyReLUD", "AddLeakyReLUD");

            if (TSCudaContext.ElementType == DType.Float16)
            {
                Logger.WriteLine($"Creating elementwise actitive kernels for Float16 type.");

                AppendTTFunc(result, "relu", "relu", DType.Float16);
                AppendTTTFunc(result, "relud", "relud", DType.Float16);
                AppendTTTTFunc(result, "addrelud", "addreludhalf", DType.Float16);

                AppendTTFunc(result, "SiLU", "SiLUHalf", DType.Float16);
                AppendTTTFunc(result, "SiLUD", "SiLUDHalf", DType.Float16);
                AppendTTTTFunc(result, "AddSiLUD", "AddSiLUDHalf", DType.Float16);


                AppendTTFunc(result, "LeakyReLU", "LeakyReLUHalf", DType.Float16);
                AppendTTTFunc(result, "LeakyReLUD", "LeakyReLUDHalf", DType.Float16);
                AppendTTTTFunc(result, "AddLeakyReLUD", "AddLeakyReLUDHalf", DType.Float16);
            }

            return result.ToString();
        }

        private static void AppendTTFunc(PermutationGenerator pg, string kernelBaseName, string func, DType elementType = DType.Float32)
        {
            pg.AddApplyT("t1_" + kernelBaseName, string.Format("*v = {0}(*v);", func), new DType[] {elementType });
            pg.AddApplyTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b);", func), new DType[] { elementType, elementType });
        }


        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func, DType elementType = DType.Float32)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b);", func), new DType[] { elementType, elementType });
            pg.AddApplyTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c);", func), new DType[] { elementType, elementType, elementType });
        }

        private static void AppendTTTTFunc(PermutationGenerator pg, string kernelBaseName, string func, DType elementType = DType.Float32)
        {
            pg.AddApplyTTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, *c);", func), new DType[] { elementType, elementType, elementType });
            pg.AddApplyTTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, *d);", func), new DType[] { elementType, elementType, elementType, elementType });
        }
    }
}
