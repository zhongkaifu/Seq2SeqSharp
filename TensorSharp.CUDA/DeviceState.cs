using AdvUtils;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using System;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Used by TSCudaContext to maintain per-device state
    /// </summary>
    public class DeviceState : IDisposable
    {
        private const int ScratchSpacePerSMStream = 4 * sizeof(float);


        public readonly CudaContext CudaContext;
        public readonly CudaDeviceProperties DeviceInfo;

        public readonly ObjectPool<CudaBlas> BlasHandles;
        // public readonly ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext> DnnHandles;

        public readonly IDeviceAllocator MemoryAllocator;
        public readonly ScratchSpace ScratchSpace;


        public DeviceState(int deviceId, float memoryUsageRatio = 0.9f, CudaMemoryDeviceAllocatorType allocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool)
        {
            CudaContext = new CudaContext(deviceId);
            DeviceInfo = CudaContext.GetDeviceInfo();
            Logger.WriteLine($"Cuda device '{deviceId}' DeviceName = '{DeviceInfo.DeviceName}' MultiProcessorCount = '{DeviceInfo.MultiProcessorCount}' MaxBlocksPerMultiProcessor = '{DeviceInfo.MaxBlocksPerMultiProcessor}' MaxThreadsPerMultiProcessor = '{DeviceInfo.MaxThreadsPerMultiProcessor}' MaxSharedMemoryPerMultiprocessor = '{DeviceInfo.MaxSharedMemoryPerMultiprocessor}' MemoryAllocatorType = '{allocatorType}'");

            BlasHandles = new ObjectPool<CudaBlas>(1, () =>
            {
                CudaContext.SetCurrent();
                return new CudaBlas();
            },
                blas => blas.Dispose());

            if (allocatorType == CudaMemoryDeviceAllocatorType.Basic)
            {
                MemoryAllocator = new BasicDeviceAllocator(CudaContext);
            }
            else if (allocatorType == CudaMemoryDeviceAllocatorType.CudaMemoryPool)
            {
                MemoryAllocator = new CudaMemoryPoolDeviceAllocator(CudaContext);
            }
            else
            {
                MemoryAllocator = new PoolingDeviceAllocator(CudaContext, memoryUsageRatio);
            }

            ScratchSpace = AllocScratchSpace(CudaContext, DeviceInfo);
        }

        public void Dispose()
        {
            BlasHandles.Dispose();
            CudaContext.Dispose();
            MemoryAllocator.Dispose();
        }

        private static ScratchSpace AllocScratchSpace(CudaContext context, CudaDeviceProperties deviceProps)
        {
            int size = ScratchSpacePerSMStream * deviceProps.MultiProcessorCount;
            ManagedCuda.BasicTypes.CUdeviceptr buffer = context.AllocateMemory(size);
            return new ScratchSpace() { size = size, buffer = buffer };
        }
    }
}
