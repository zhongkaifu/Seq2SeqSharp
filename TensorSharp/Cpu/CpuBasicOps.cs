using System;
using System.Reflection;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuBasicOps
    {
        public CpuBasicOps()
        {
        }


        [RegisterOpStorageType("dot", typeof(CpuStorage))]
        public Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Dot(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Mul_M_V(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return MatrixMultiplication.Mul_M_M(result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CpuStorage))]
        public Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            //Console.WriteLine($"src0 = {src.Sizes[0]}, src1 = {src.Sizes[1]}, m1_0 = {m1.Sizes[0]}, m1_1 = {m1.Sizes[1]}, m2_0 = {m2.Sizes[0]}, m2_1 = {m2.Sizes[1]}");

            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException("All tensors must have the same element type");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", "result");
            }

            if (!(m1.Storage is CpuStorage))
            {
                throw new ArgumentException("m1 must be a CPU tensor", "m1");
            }

            if (!(m2.Storage is CpuStorage))
            {
                throw new ArgumentException("m2 must be a CPU tensor", "m2");
            }

            if (src.DimensionCount != 2)
            {
                throw new ArgumentException("src must be a matrix", "src");
            }

            if (m1.DimensionCount != 2)
            {
                throw new ArgumentException("m1 must be a matrix", "m1");
            }

            if (m2.DimensionCount != 2)
            {
                throw new ArgumentException("m2 must be a matrix", "m2");
            }

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
            {
                throw new InvalidOperationException("Size mismatch");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }


            MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);


            return writeTarget;
        }

        [RegisterOpStorageType("addmmbatch", typeof(CpuStorage))]
        public Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException("All tensors must have the same element type");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", "result");
            }

            if (!(m1.Storage is CpuStorage))
            {
                throw new ArgumentException("m1 must be a CPU tensor", "m1");
            }

            if (!(m2.Storage is CpuStorage))
            {
                throw new ArgumentException("m2 must be a CPU tensor", "m2");
            }

            if (src.DimensionCount != 3)
            {
                throw new ArgumentException("src must be a matrix", "src");
            }

            if (m1.DimensionCount != 3)
            {
                throw new ArgumentException("m1 must be a matrix", "m1");
            }

            if (m2.DimensionCount != 3)
            {
                throw new ArgumentException("m2 must be a matrix", "m2");
            }

            if (src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            int batchSize = (int)src.Sizes[0];
            for (int i = 0; i < batchSize; i++)
            {
                Tensor a = m1.Select(0, i);// m1.Narrow(0, i, 1).View(m1.Sizes[1], m1.Sizes[2]);
                Tensor b = m2.Select(0, i); // m2.Narrow(0, i, 1).View(m2.Sizes[1], m2.Sizes[2]);
                Tensor r = writeTarget.Select(0, i); // writeTarget.Narrow(0, i, 1).View(writeTarget.Sizes[1], writeTarget.Sizes[2]);

                MatrixMultiplication.Gemm(alpha, a, b, beta, r);
            }


            //MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);


            return writeTarget;
        }



        private readonly MethodInfo abs_func = NativeWrapper.GetMethod("TS_Abs");
        [RegisterOpStorageType("abs", typeof(CpuStorage))]
        public Tensor Abs(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(abs_func, result, src); }

        private readonly MethodInfo neg_func = NativeWrapper.GetMethod("TS_Neg");
        [RegisterOpStorageType("neg", typeof(CpuStorage))]
        public Tensor Neg(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(neg_func, result, src); }

        private readonly MethodInfo sign_func = NativeWrapper.GetMethod("TS_Sign");
        [RegisterOpStorageType("sign", typeof(CpuStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sign_func, result, src); }


        private readonly MethodInfo sqrt_func = NativeWrapper.GetMethod("TS_Sqrt");
        [RegisterOpStorageType("sqrt", typeof(CpuStorage))]
        public Tensor Sqrt(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sqrt_func, result, src); }


        private readonly MethodInfo rsqrt_func = NativeWrapper.GetMethod("TS_Rsqrt");
        [RegisterOpStorageType("rsqrt", typeof(CpuStorage))]
        public Tensor Rsqrt(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(rsqrt_func, result, src); }

        private readonly MethodInfo exp_func = NativeWrapper.GetMethod("TS_Exp");
        [RegisterOpStorageType("exp", typeof(CpuStorage))]
        public Tensor Exp(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(exp_func, result, src); }

        private readonly MethodInfo log_func = NativeWrapper.GetMethod("TS_Log");
        [RegisterOpStorageType("log", typeof(CpuStorage))]
        public Tensor Log(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log_func, result, src); }

        private readonly MethodInfo log1p_func = NativeWrapper.GetMethod("TS_Log1p");
        [RegisterOpStorageType("log1p", typeof(CpuStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log1p_func, result, src); }

        private readonly MethodInfo floor_func = NativeWrapper.GetMethod("TS_Floor");
        [RegisterOpStorageType("floor", typeof(CpuStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(floor_func, result, src); }

        private readonly MethodInfo ceil_func = NativeWrapper.GetMethod("TS_Ceil");
        [RegisterOpStorageType("ceil", typeof(CpuStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(ceil_func, result, src); }

        private readonly MethodInfo round_func = NativeWrapper.GetMethod("TS_Round");
        [RegisterOpStorageType("round", typeof(CpuStorage))]
        public Tensor Round(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(round_func, result, src); }

        private readonly MethodInfo trunc_func = NativeWrapper.GetMethod("TS_Trunc");
        [RegisterOpStorageType("trunc", typeof(CpuStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(trunc_func, result, src); }

        private readonly MethodInfo frac_func = NativeWrapper.GetMethod("TS_Frac");
        [RegisterOpStorageType("frac", typeof(CpuStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(frac_func, result, src); }


        private readonly MethodInfo relu_func = NativeWrapper.GetMethod("TS_Relu");
        [RegisterOpStorageType("relu", typeof(CpuStorage))]
        public Tensor Relu(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(relu_func, result, src); }


        private readonly MethodInfo sin_func = NativeWrapper.GetMethod("TS_Sin");
        [RegisterOpStorageType("sin", typeof(CpuStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sin_func, result, src); }

        private readonly MethodInfo cos_func = NativeWrapper.GetMethod("TS_Cos");
        [RegisterOpStorageType("cos", typeof(CpuStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cos_func, result, src); }

        private readonly MethodInfo tan_func = NativeWrapper.GetMethod("TS_Tan");
        [RegisterOpStorageType("tan", typeof(CpuStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tan_func, result, src); }


        private readonly MethodInfo asin_func = NativeWrapper.GetMethod("TS_Asin");
        [RegisterOpStorageType("asin", typeof(CpuStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(asin_func, result, src); }

        private readonly MethodInfo acos_func = NativeWrapper.GetMethod("TS_Acos");
        [RegisterOpStorageType("acos", typeof(CpuStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(acos_func, result, src); }

        private readonly MethodInfo atan_func = NativeWrapper.GetMethod("TS_Atan");
        [RegisterOpStorageType("atan", typeof(CpuStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(atan_func, result, src); }


        private readonly MethodInfo sinh_func = NativeWrapper.GetMethod("TS_Sinh");
        [RegisterOpStorageType("sinh", typeof(CpuStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sinh_func, result, src); }

        private readonly MethodInfo cosh_func = NativeWrapper.GetMethod("TS_Cosh");
        [RegisterOpStorageType("cosh", typeof(CpuStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cosh_func, result, src); }

        private readonly MethodInfo tanh_func = NativeWrapper.GetMethod("TS_Tanh");
        [RegisterOpStorageType("tanh", typeof(CpuStorage))]
        public Tensor Tanh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, src); }


        private readonly MethodInfo sigmoid_func = NativeWrapper.GetMethod("TS_Sigmoid");
        [RegisterOpStorageType("sigmoid", typeof(CpuStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sigmoid_func, result, src); }

        private readonly MethodInfo tanhD_func = NativeWrapper.GetMethod("TS_TanhD");
        [RegisterOpStorageType("tanhD", typeof(CpuStorage))]
        public Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return NativeWrapper.InvokeNullableResultElementwise(tanhD_func, result, resW, resG); }

        private readonly MethodInfo sigmoidD_func = NativeWrapper.GetMethod("TS_SigmoidD");
        [RegisterOpStorageType("sigmoidD", typeof(CpuStorage))]
        public Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) { return NativeWrapper.InvokeNullableResultElementwise(sigmoidD_func, result, resW, resG); }

        private readonly MethodInfo add3_func = NativeWrapper.GetMethod("TS_Add3");
        [RegisterOpStorageType("add3", typeof(CpuStorage))]
        public Tensor Add3(Tensor result, Tensor x, Tensor y, Tensor z) { return NativeWrapper.InvokeNullableResultElementwise(add3_func, result, x, y, z); }

        private readonly MethodInfo add4_func = NativeWrapper.GetMethod("TS_Add4");
        [RegisterOpStorageType("add4", typeof(CpuStorage))]
        public Tensor Add4(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return NativeWrapper.InvokeNullableResultElementwise(add4_func, result, x, y, z, w); }


        private readonly MethodInfo addmul_func = NativeWrapper.GetMethod("TS_AddMul");
        [RegisterOpStorageType("addmul", typeof(CpuStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return NativeWrapper.InvokeNullableResultElementwise(addmul_func, result, x, y, z); }

        private readonly MethodInfo addmulv_func = NativeWrapper.GetMethod("TS_AddMulV");
        [RegisterOpStorageType("addmulv", typeof(CpuStorage))]
        public Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return NativeWrapper.InvokeNullableResultElementwise(addmulv_func, result, x, y, z); }


        private readonly MethodInfo maskfill_func = NativeWrapper.GetMethod("TS_MaskFill");
        [RegisterOpStorageType("maskfill", typeof(CpuStorage))]
        public Tensor MaskFill(Tensor result, Tensor t, Tensor mask, float defValue) { return NativeWrapper.InvokeNullableResultElementwise(maskfill_func, result, t, mask, defValue); }


        private readonly MethodInfo atan2_func = NativeWrapper.GetMethod("TS_Atan2");
        [RegisterOpStorageType("atan2", typeof(CpuStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return NativeWrapper.InvokeNullableResultElementwise(atan2_func, result, srcY, srcX); }

        private readonly MethodInfo pow_func = NativeWrapper.GetMethod("TS_Pow");
        [RegisterOpStorageType("pow", typeof(CpuStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value) { return NativeWrapper.InvokeNullableResultElementwise(pow_func, result, src, value); }

        private readonly MethodInfo tpow_func = NativeWrapper.GetMethod("TS_Tpow");
        [RegisterOpStorageType("tpow", typeof(CpuStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tpow_func, result, value, src); }

        private readonly MethodInfo lerp_func = NativeWrapper.GetMethod("TS_Lerp");
        [RegisterOpStorageType("lerp", typeof(CpuStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, srcA, srcB, weight); }

        private readonly MethodInfo clamp_func = NativeWrapper.GetMethod("TS_Clamp");
        [RegisterOpStorageType("clamp", typeof(CpuStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max) { return NativeWrapper.InvokeNullableResultElementwise(clamp_func, result, src, min, max); }


        private readonly MethodInfo mulmuladd_func = NativeWrapper.GetMethod("TS_MulMulAdd");
        [RegisterOpStorageType("mulmuladd", typeof(CpuStorage))]
        public Tensor MulMulAdd(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW) { return NativeWrapper.InvokeNullableResultElementwise(mulmuladd_func, result, srcX, srcY, srcZ, srcW); }



        private readonly MethodInfo addtanh_func = NativeWrapper.GetMethod("TS_AddTanh");
        [RegisterOpStorageType("addtanh", typeof(CpuStorage))]
        public Tensor AddTanh(Tensor result, Tensor srcX, Tensor srcY) { return NativeWrapper.InvokeNullableResultElementwise(addtanh_func, result, srcX, srcY); }


        private readonly MethodInfo addtanh3_func = NativeWrapper.GetMethod("TS_AddTanh3");
        [RegisterOpStorageType("addtanh3", typeof(CpuStorage))]
        public Tensor AddTanh3(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ) { return NativeWrapper.InvokeNullableResultElementwise(addtanh3_func, result, srcX, srcY, srcZ); }


        private readonly MethodInfo addtanhD_func = NativeWrapper.GetMethod("TS_AddTanhD");
        [RegisterOpStorageType("addtanhD", typeof(CpuStorage))]
        public Tensor AddTanhD(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ) { return NativeWrapper.InvokeNullableResultElementwise(addtanhD_func, result, srcX, srcY, srcZ); }


        private readonly MethodInfo addreluD_func = NativeWrapper.GetMethod("TS_AddReluD");
        [RegisterOpStorageType("addrelud", typeof(CpuStorage))]
        public Tensor AddReluD(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ) { return NativeWrapper.InvokeNullableResultElementwise(addreluD_func, result, srcX, srcY, srcZ); }

        private readonly MethodInfo add_func = NativeWrapper.GetMethod("TS_Add");
        [RegisterOpStorageType("addv", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(add_func, result, lhs, rhs); }

        private readonly MethodInfo sub_func = NativeWrapper.GetMethod("TS_Sub");
        [RegisterOpStorageType("subv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(sub_func, result, lhs, rhs); }

        private readonly MethodInfo rsub_func = NativeWrapper.GetMethod("TS_Rsub");
        [RegisterOpStorageType("rsubv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rsub_func, result, rhs, lhs); }

        private readonly MethodInfo mul_func = NativeWrapper.GetMethod("TS_Mul");
        [RegisterOpStorageType("mulv", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mul_func, result, lhs, rhs); }

        private readonly MethodInfo div_func = NativeWrapper.GetMethod("TS_Div");
        [RegisterOpStorageType("divv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(div_func, result, lhs, rhs); }

        private readonly MethodInfo rdiv_func = NativeWrapper.GetMethod("TS_Rdiv");
        [RegisterOpStorageType("rdivv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rdiv_func, result, rhs, lhs); }

        private readonly MethodInfo mod_func = NativeWrapper.GetMethod("TS_Mod");
        [RegisterOpStorageType("modv", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mod_func, result, lhs, rhs); }


        private readonly MethodInfo gtValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("gtValue", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtValue_func, result, lhs, rhs); }

        private readonly MethodInfo ltValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("ltValue", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltValue_func, result, lhs, rhs); }

        private readonly MethodInfo geValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("geValue", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(geValue_func, result, lhs, rhs); }

        private readonly MethodInfo leValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("leValue", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(leValue_func, result, lhs, rhs); }

        private readonly MethodInfo eqValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("eqValue", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqValue_func, result, lhs, rhs); }

        private readonly MethodInfo neValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("neValue", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(neValue_func, result, lhs, rhs); }



        private readonly MethodInfo cadd_func = NativeWrapper.GetMethod("TS_CAdd");
        [RegisterOpStorageType("addt", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cadd_func, result, lhs, rhs); }

        private readonly MethodInfo csub_func = NativeWrapper.GetMethod("TS_CSub");
        [RegisterOpStorageType("subt", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(csub_func, result, lhs, rhs); }

        private readonly MethodInfo cmul_func = NativeWrapper.GetMethod("TS_CMul");
        [RegisterOpStorageType("mult", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmul_func, result, lhs, rhs); }

        private readonly MethodInfo cdiv_func = NativeWrapper.GetMethod("TS_CDiv");
        [RegisterOpStorageType("divt", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cdiv_func, result, lhs, rhs); }

        private readonly MethodInfo cmod_func = NativeWrapper.GetMethod("TS_CMod");
        [RegisterOpStorageType("modt", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmod_func, result, lhs, rhs); }


        private readonly MethodInfo gtTensor_func = NativeWrapper.GetMethod("TS_gtTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtTensor_func, result, lhs, rhs); }

        private readonly MethodInfo ltTensor_func = NativeWrapper.GetMethod("TS_ltTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltTensor_func, result, lhs, rhs); }

        private readonly MethodInfo geTensor_func = NativeWrapper.GetMethod("TS_geTensor");
        [RegisterOpStorageType("geTensor", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(geTensor_func, result, lhs, rhs); }

        private readonly MethodInfo leTensor_func = NativeWrapper.GetMethod("TS_leTensor");
        [RegisterOpStorageType("leTensor", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(leTensor_func, result, lhs, rhs); }

        private readonly MethodInfo eqTensor_func = NativeWrapper.GetMethod("TS_eqTensor");
        [RegisterOpStorageType("eqTensor", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqTensor_func, result, lhs, rhs); }

        private readonly MethodInfo neTensor_func = NativeWrapper.GetMethod("TS_neTensor");
        [RegisterOpStorageType("neTensor", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(neTensor_func, result, lhs, rhs); }


        private readonly MethodInfo sum_func = NativeWrapper.GetMethod("TS_Sum");
        [RegisterOpStorageType("sum", typeof(CpuStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(sum_func, result, src, dimension); }

        private readonly MethodInfo prod_func = NativeWrapper.GetMethod("TS_Prod");
        [RegisterOpStorageType("prod", typeof(CpuStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(prod_func, result, src, dimension); }

        private readonly MethodInfo min_func = NativeWrapper.GetMethod("TS_Min");
        [RegisterOpStorageType("min", typeof(CpuStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(min_func, result, src, dimension); }

        private readonly MethodInfo max_func = NativeWrapper.GetMethod("TS_Max");
        [RegisterOpStorageType("max", typeof(CpuStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(max_func, result, src, dimension); }


        private readonly MethodInfo argmin_func = NativeWrapper.GetMethod("TS_Argmin");
        [RegisterOpStorageType("argmin", typeof(CpuStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }

        private readonly MethodInfo argmax_func = NativeWrapper.GetMethod("TS_Argmax");
        [RegisterOpStorageType("argmax", typeof(CpuStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }



        private readonly MethodInfo mean_func = NativeWrapper.GetMethod("TS_Mean");
        [RegisterOpStorageType("mean", typeof(CpuStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(mean_func, result, src, dimension); }

        private readonly MethodInfo norm_func = NativeWrapper.GetMethod("TS_Norm");
        [RegisterOpStorageType("norm", typeof(CpuStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return NativeWrapper.InvokeNullableResultDimensionwise(norm_func, result, src, dimension, value); }

        private readonly MethodInfo std_func = NativeWrapper.GetMethod("TS_Std");
        [RegisterOpStorageType("std", typeof(CpuStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(std_func, result, src, dimension, normByN); }

        private readonly MethodInfo var_func = NativeWrapper.GetMethod("TS_Var");
        [RegisterOpStorageType("var", typeof(CpuStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(var_func, result, src, dimension, normByN); }



        private readonly MethodInfo sumall_func = NativeWrapper.GetMethod("TS_SumAll");
        [RegisterOpStorageType("sumall", typeof(CpuStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(sumall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo prodall_func = NativeWrapper.GetMethod("TS_ProdAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(prodall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo minall_func = NativeWrapper.GetMethod("TS_MinAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(minall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo maxall_func = NativeWrapper.GetMethod("TS_MaxAll");
        [RegisterOpStorageType("maxall", typeof(CpuStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(maxall_func, writeTarget, src);
            return writeTarget;
        }


        private readonly MethodInfo meanall_func = NativeWrapper.GetMethod("TS_MeanAll");
        [RegisterOpStorageType("meanall", typeof(CpuStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(meanall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo varall_func = NativeWrapper.GetMethod("TS_VarAll");
        [RegisterOpStorageType("varall", typeof(CpuStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(varall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo stdall_func = NativeWrapper.GetMethod("TS_StdAll");
        [RegisterOpStorageType("stdall", typeof(CpuStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(stdall_func, writeTarget, src);
            return writeTarget;
        }


        private readonly MethodInfo layerNorm_func = NativeWrapper.GetMethod("TS_LayerNorm");
        [RegisterOpStorageType("layernorm", typeof(CpuStorage))]
        public Tensor LayerNorm(Tensor result, Tensor src, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            NativeWrapper.InvokeTypeMatch(layerNorm_func, writeTarget, src, gamma_, beta_, eps, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        private readonly MethodInfo layerNormGrad_func = NativeWrapper.GetMethod("TS_LayerNormGrad");
        [RegisterOpStorageType("layernormgrad", typeof(CpuStorage))]
        public Tensor LayerNormGrad(Tensor result, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x_, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, adj_, false, adj_.Sizes);
            NativeWrapper.InvokeTypeMatch(layerNormGrad_func, writeTarget, gradGamma_, gradBeta_, adj_, y_, x_, gamma_, beta_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], eps);
            return writeTarget;
        }



        private readonly MethodInfo addlayerNorm_func = NativeWrapper.GetMethod("TS_AddLayerNorm");
        [RegisterOpStorageType("addlayernorm", typeof(CpuStorage))]
        public Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src1, false, src1.Sizes);
            NativeWrapper.InvokeTypeMatch(addlayerNorm_func, writeTarget, src1, src2, gamma_, beta_, eps, (int)src1.Sizes[0], (int)src1.Sizes[1]);
            return writeTarget;
        }

        private readonly MethodInfo addlayerNormGrad_func = NativeWrapper.GetMethod("TS_AddLayerNormGrad");
        [RegisterOpStorageType("addlayernormgrad", typeof(CpuStorage))]
        public void AddLayerNormGrad(Tensor result1, Tensor result2, Tensor gradGamma_, Tensor gradBeta_, Tensor adj_, Tensor y_, Tensor x1_, Tensor x2_, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget1 = TensorResultBuilder.GetWriteTarget(result1, adj_, false, adj_.Sizes);
            Tensor writeTarget2 = TensorResultBuilder.GetWriteTarget(result2, adj_, false, adj_.Sizes);
            NativeWrapper.InvokeTypeMatch(addlayerNormGrad_func, writeTarget1, writeTarget2, gradGamma_, gradBeta_, adj_, y_, x1_, x2_, gamma_, beta_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], eps);
        }

        private readonly MethodInfo softmax_func = NativeWrapper.GetMethod("TS_Softmax");
        [RegisterOpStorageType("softmax", typeof(CpuStorage))]
        public Tensor Softmax(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            NativeWrapper.InvokeTypeMatch(softmax_func, writeTarget, src, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        private readonly MethodInfo softmaxmask_func = NativeWrapper.GetMethod("TS_SoftmaxMask");
        [RegisterOpStorageType("softmaxmask", typeof(CpuStorage))]
        public Tensor SoftmaxMask(Tensor result, Tensor src, Tensor mask)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            NativeWrapper.InvokeTypeMatch(softmaxmask_func, writeTarget, src, mask, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }


        private readonly MethodInfo softmaxGrad_func = NativeWrapper.GetMethod("TS_SoftmaxGrad");
        [RegisterOpStorageType("softmaxgrad", typeof(CpuStorage))]
        public Tensor SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_, bool addGrad = true)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(grad_, adj_, false, adj_.Sizes);
            NativeWrapper.InvokeTypeMatch(softmaxGrad_func, writeTarget, adj_, val_, (int)adj_.Sizes[0], (int)adj_.Sizes[1], addGrad);
            return writeTarget;
        }


        private readonly MethodInfo rmsProp_func = NativeWrapper.GetMethod("TS_RMSProp");
        [RegisterOpStorageType("rmsprop", typeof(CpuStorage))]
        public Tensor RMSProp(Tensor tw, Tensor tg, Tensor tc, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            NativeWrapper.InvokeTypeMatch(rmsProp_func, tw, tg, tc, (int)tw.Sizes[0], (int)tw.Sizes[1], batchSize, step_size, clipval, regc, decay_rate, eps);
            return tw;
        }

        private readonly MethodInfo adam_func = NativeWrapper.GetMethod("TS_Adam");
        [RegisterOpStorageType("adam", typeof(CpuStorage))]
        public Tensor Adam(Tensor tw, Tensor tg, Tensor tv, Tensor tm, int batchSize, float step_size, float clipval, float regc, float decay_rate_v, float decay_rate_m, int iter, float eps)
        {
            NativeWrapper.InvokeTypeMatch(adam_func, tw, tg, tv, tm, (int)tw.Sizes[0], (int)tw.Sizes[1], batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, eps);
            return tw;
        }

        private readonly MethodInfo normall_func = NativeWrapper.GetMethod("TS_NormAll");
        [RegisterOpStorageType("normall", typeof(CpuStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(normall_func, writeTarget, src, value);
            return writeTarget;
        }
    }
}
