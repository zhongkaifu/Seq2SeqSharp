namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class CudaReduceKernels : CudaCode
    {
        public CudaReduceKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math")
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
