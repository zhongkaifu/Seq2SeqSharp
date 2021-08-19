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

        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        public void Fill(Tensor result, float value)
        {
            TensorApplyCPU.Fill(result, value);
        }


        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        public void Copy(Tensor result, Tensor src)
        {
            if (result.ElementCount() != src.ElementCount())
            {
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            }

            TensorApplyCPU.Copy(result, src);
        }

    }
}
