using System;
using System.Reflection;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuFillCopyOps
    {
        public CpuFillCopyOps()
        {
        }


        private readonly MethodInfo fill_func = NativeWrapper.GetMethod("TS_Fill");
        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        public void Fill(Tensor result, float value)
        {
            NativeWrapper.InvokeTypeMatch(fill_func, result, value);
        }


        private readonly MethodInfo copy_func = NativeWrapper.GetMethod("TS_Copy");
        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        public void Copy(Tensor result, Tensor src)
        {
            if (result.ElementCount() != src.ElementCount())
            {
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            }

            NativeWrapper.Invoke(copy_func, result, src);
        }
    }
}
