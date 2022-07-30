using AdvUtils;
using System;

namespace TensorSharp.Cpu
{
    public class CpuAllocator : IAllocator
    {
        private BlasEnum m_blasEnum;
        public BlasEnum BlasEnum => m_blasEnum;
        public CpuAllocator(BlasEnum blasEnum)
        {
            m_blasEnum = blasEnum;
            if (m_blasEnum == BlasEnum.MKL)
            {
                Logger.WriteLine("Setting environment variable for MKL runtime.");
                Environment.SetEnvironmentVariable("MKL_ENABLE_INSTRUCTIONS", "AVX2");
            }
        }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CpuStorage(this, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
