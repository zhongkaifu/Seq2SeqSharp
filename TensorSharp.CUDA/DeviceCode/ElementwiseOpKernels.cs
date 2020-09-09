namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseOpKernels : CudaCode
    {
        public ElementwiseOpKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math")
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


            return result.ToString();
        }

        private static void AppendTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, b);", func));
            pg.AddApplyTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, c);", func));
        }


        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b);", func));
            pg.AddApplyTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c);", func));
        }
    }
}
