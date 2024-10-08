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
    public class ElementwiseOpKernels : CudaCode
    {
        public ElementwiseOpKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            PermutationGenerator result = new PermutationGenerator();
       
            AppendTTSFunc(result, "add", "add_op");
            AppendTTSFunc(result, "sub", "sub_op");
            AppendTTSFunc(result, "rsub", "rsub_op");
            AppendTTSFunc(result, "mul", "mul_op");
           
            AppendTTSFunc(result, "div", "div_op");           
            AppendTTSFunc(result, "rdiv", "rdiv_op");
            AppendTTSFunc(result, "mod", "Mod_op");

            AppendTTSFunc(result, "gt", "gt_op");
            AppendTTSFunc(result, "lt", "lt_op");
            AppendTTSFunc(result, "ge", "gt_op");
            AppendTTSFunc(result, "le", "le_op");
            AppendTTSFunc(result, "eq", "eq_op");
            AppendTTSFunc(result, "ne", "ne_op");

            AppendTTTFunc(result, "cadd", "add_op");           
            AppendTTTFunc(result, "csub", "sub_op");
            AppendTTTFunc(result, "cmul", "mul_op");
            AppendTTTFunc(result, "cdiv", "div_op");
            AppendTTTFunc(result, "cmod", "Mod_op");

            AppendTTTFunc(result, "cgt", "gt_op");
            AppendTTTFunc(result, "clt", "lt_op");
            AppendTTTFunc(result, "cge", "gt_op");
            AppendTTTFunc(result, "cle", "le_op");
            AppendTTTFunc(result, "ceq", "eq_op");
            AppendTTTFunc(result, "cne", "ne_op");

            AppendAtomicAdd(result, "atomicAdd", DType.Float32);
            

            if (TSCudaContext.ElementType == DType.Float16)
            {
                Logger.WriteLine(Logger.Level.debug, $"Creating elementwise kernels for Float16 type.");

                AppendTTSHalfFunc(result, "add", "__hadd");
                AppendTTSHalfFunc(result, "mul", "__hmul");
                AppendTTSHalfFunc(result, "div", "__hdiv");
                AppendTTTFunc(result, "cadd", "__hadd", DType.Float16);
                AppendTTTFunc(result, "cmul", "__hmul", DType.Float16);
                AppendTTTFunc(result, "cdiv", "__hdiv", DType.Float16);

                AppendAtomicAdd(result, "atomicAdd", DType.Float16);
            }

                return result.ToString();
        }


        private static void AppendAtomicAdd(PermutationGenerator pg, string kernelBaseName, DType elementType = DType.Float32)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, "atomicAdd(a, *b);", new DType[] { elementType, elementType });
        }


        private static void AppendTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, b);", func));
            pg.AddApplyTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, c);", func));
        }

        private static void AppendTTSHalfFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTSHalf("t1_" + kernelBaseName, string.Format("*a = {0}(*a, b);", func));
            pg.AddApplyTTSHalf("t2_" + kernelBaseName, string.Format("*a = {0}(*b, c);", func));
        }

        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func, DType elementType = DType.Float32)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b);", func), new DType[] {elementType, elementType });
            pg.AddApplyTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c);", func), new DType[] {elementType, elementType, elementType });
        }
    }
}
