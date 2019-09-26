using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{
    public static class TensorAllocator
    {
        private static IAllocator[] allocator = null;
        private static TSCudaContext cudaContext = null;
        private static int[] deviceIds;
        private static ArchTypeEnums m_archType;


        public static void InitDevices(ArchTypeEnums archType, int[] ids)
        {
            m_archType = archType;
            if (m_archType == ArchTypeEnums.GPU)
            {
                deviceIds = ids;

                foreach (var id in deviceIds)
                {
                    Logger.WriteLine($"Initialize device '{id}'");
                }

                cudaContext = new TSCudaContext(deviceIds);
                cudaContext.Precompile(Console.Write);
                cudaContext.CleanUnusedPTX();

                allocator = new IAllocator[deviceIds.Length];
            }
        }

        public static IAllocator Allocator(int deviceId)
        {
            if (m_archType == ArchTypeEnums.GPU)
            {
                int idx = GetDeviceIdIndex(deviceId);
                if (allocator[idx] == null)
                {
                    allocator[idx] = new CudaAllocator(cudaContext, deviceId);
                }

                return allocator[idx];
            }
            else
            {
                return new CpuAllocator();
            }
        }

        private static int GetDeviceIdIndex(int id)
        {
            for (int i = 0; i < deviceIds.Length; i++)
            {
                if (deviceIds[i] == id)
                {
                    return i;
                }
            }

            return -1;
        }

        public static void FreeMemoryAllDevices(bool callGC = false)
        {
            GC.Collect();
            if (cudaContext != null)
            {
                cudaContext.FreeMemoryAllDevices(callGC);
            }
        }
    }
}
