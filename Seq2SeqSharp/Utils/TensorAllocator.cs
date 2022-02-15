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

        public static void InitDevices(ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f, string[] compilerOptions = null)
        {
            m_archType = archType;
            m_deviceIds = ids;
            m_allocator = new IAllocator[m_deviceIds.Length];

            if (m_archType == ProcessorTypeEnums.GPU)
            {
                foreach (int id in m_deviceIds)
                {
                    Logger.WriteLine($"Initialize device '{id}'");
                }

                m_cudaContext = new TSCudaContext(m_deviceIds, memoryUsageRatio, compilerOptions);
                m_cudaContext.Precompile(Console.Write);
                m_cudaContext.CleanUnusedPTX();
            }
        }

        public static IAllocator Allocator(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            if (m_archType == ProcessorTypeEnums.GPU)
            {
                if (m_allocator[idx] == null)
                {
                    m_allocator[idx] = new CudaAllocator(m_cudaContext, deviceId);
                }
            }
            else
            {
                if (m_allocator[idx] == null)
                {
                    m_allocator[idx] = new CpuAllocator();
                }
            }

            return m_allocator[idx];
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

            string strIds = string.Empty;
            foreach (var item in m_deviceIds)
            {
                strIds = strIds + " " + item.ToString();
            }

            throw new ArgumentException($"Failed to get deviceId '{id}', deviceId List = {strIds}");
        }
    }
}
