using AdvUtils;
using System;
using TensorSharp;
using TensorSharp.Cpu;

namespace Seq2SeqSharp
{
    public static class TensorAllocator
    {
        private static IAllocator[] m_allocator = null;
        private static ProcessorTypeEnums m_archType;


        public static void InitDevices(ProcessorTypeEnums archType, int[] ids)
        {
            m_archType = archType;
            if (m_archType == ProcessorTypeEnums.GPU)
            {
                throw new NotSupportedException("GPU is not supported for .NET Core");
            }
            else
            {
                m_allocator = new IAllocator[1];
            }
        }

        public static IAllocator Allocator(int deviceId)
        {
            if (m_archType == ProcessorTypeEnums.GPU)
            {
                throw new NotSupportedException("GPU is not supported for .NET Core");
            }
            else
            {
                if (m_allocator[0] == null)
                {
                    m_allocator[0] = new CpuAllocator();
                }

                return m_allocator[0];
            }
        }

        public static void FreeMemoryAllDevices(bool callGC = false)
        {
            GC.Collect();
        }
    }
}
