namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseTriKernels : CudaCode
    {
        public ElementwiseTriKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "ApplyMacros")
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
