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
    public class ElementwiseKernels : CudaCode
    {
        public ElementwiseKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "Fp16")
        {
        }

        private static string GetFullCode()
        {
            PermutationGenerator result = new PermutationGenerator();
            AppendTTFunc(result, "abs", "fabs");
            AppendTTFunc(result, "neg", "-");
            AppendTTFunc(result, "sign", "sgn");

            AppendTTFunc(result, "sqrt", "sqrtf");
            AppendTTFunc(result, "rsqrt", "rsqrtf");

            AppendTTFunc(result, "exp", "expf");
            AppendTTFunc(result, "log", "logf");
            AppendTTFunc(result, "log1p", "log1p");
            AppendTTFunc(result, "floor", "floor");
            AppendTTFunc(result, "ceil", "ceil");
            AppendTTFunc(result, "round", "round");
            AppendTTFunc(result, "trunc", "trunc");
            AppendTTFunc(result, "frac", "Frac");

            AppendTTTTTFunc(result, "mulmuladd", "MulMulAdd");
            AppendTTTTFunc(result, "addmul", "AddMul");

            //x + y * z
            AppendTTTSFunc(result, "addmulv", "AddMul");
            AppendTTTTFunc(result, "adddiv", "AddDiv");
            AppendTTTSFunc(result, "maskfill", "MaskFill");

            result.AddApplyTS("t1_pow", "*a = powf(*a, b);");
            result.AddApplyTTS("t2_pow", "*a = powf(*b, c);");
            result.AddApplyTS("t1_tpow", "*a = powf(b, *a);");
            result.AddApplyTTS("t2_tpow", "*a = powf(c, *b);");

            result.AddApplyTTTS("lerp", "*a = Lerp(*b, *c, d);");

            result.AddApplyTSS("t1_clamp", "*a = Clamp(*a, b, c);");
            result.AddApplyTTSS("t2_clamp", "*a = Clamp(*b, c, d);");


            if (TSCudaContext.ElementType == DType.Float16)
            {
                Logger.WriteLine($"Creating elementwise kernels for Float16 type.");

                result.AddApplyTTSHalf("t1_addmulv", "*a = __hadd(*a, __hmul(*b, c));");
                result.AddApplyTTTSHalf("t2_addmulv", "*a = __hadd(*b, __hmul(*c, d));");
                result.AddApplyTT("t2_float2half", "*a = __float2half(*b);", new DType[] { DType.Float16, DType.Float32 });
                result.AddApplyTT("t2_half2float", "*a = __half2float(*b);", new DType[] { DType.Float32, DType.Float16 });
            }

            return result.ToString();
        }

        private static void AppendTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyT("t1_" + kernelBaseName, string.Format("*v = {0}(*v);", func));
            pg.AddApplyTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b);", func));
        }

        private static void AppendTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, b);", func));
            pg.AddApplyTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, c);", func));
        }

        private static void AppendTTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, c);", func));
            pg.AddApplyTTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, d);", func));
        }

        //private static void AppendTTTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        //{
        //    pg.AddApplyTTTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, *c, d);", func));
        //    pg.AddApplyTTTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, *d, e);", func));
        //}

        //private static void AppendTTTTSSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        //{
        //    pg.AddApplyTTTSS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, *c, d, e);", func));
        //    pg.AddApplyTTTTSS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, *d, e, f);", func));
        //}

        //private static void AppendTTTSSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        //{
        //    pg.AddApplyTTSS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, c, d);", func));
        //    pg.AddApplyTTTSS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, d, e);", func));
        //}

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

        private static void AppendTTTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTTTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b, *c, *d);", func));
            pg.AddApplyTTTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c, *d, *e);", func));
        }
    }
}
