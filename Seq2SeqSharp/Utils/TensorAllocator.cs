using AdvUtils;
using System;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{
    public static class TensorAllocator
    {
        private static IAllocator[] m_allocator = null;
        private static TSCudaContext m_cudaContext = null;
        private static int[] m_deviceIds;
        private static ProcessorTypeEnums m_archType;


        public static void InitDevices(ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f)
        {
            m_archType = archType;
            if (m_archType == ProcessorTypeEnums.GPU)
            {
                m_deviceIds = ids;

                foreach (int id in m_deviceIds)
                {
                    Logger.WriteLine($"Initialize device '{id}'");
                }

                m_cudaContext = new TSCudaContext(m_deviceIds, memoryUsageRatio);
                m_cudaContext.Precompile(Console.Write);
                m_cudaContext.CleanUnusedPTX();

                m_allocator = new IAllocator[m_deviceIds.Length];
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
                int idx = GetDeviceIdIndex(deviceId);
                if (m_allocator[idx] == null)
                {
                    m_allocator[idx] = new CudaAllocator(m_cudaContext, deviceId);
                }

                return m_allocator[idx];
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

        private static int GetDeviceIdIndex(int id)
        {
            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                if (m_deviceIds[i] == id)
                {
                    return i;
                }
            }

            return -1;
        }
    }
}
