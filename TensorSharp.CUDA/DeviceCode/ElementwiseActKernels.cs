namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseActKernels : CudaCode
    {
        public ElementwiseActKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math")
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


            AppendTTFunc(result, "Swish", "Swish");
            AppendTTTFunc(result, "SwishD", "SwishD");
            AppendTTTTFunc(result, "AddSwishD", "AddSwishD");


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
