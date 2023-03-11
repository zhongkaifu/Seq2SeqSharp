using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;

namespace TensorSharp.CUDA.ContextState
{
    /// <summary>
    /// This allocator simply forwards all alloc/free requests to CUDA. This will generally be slow
    /// because calling cudaMalloc causes GPU synchronization
    /// </summary>
    public class CudaMemoryPoolDeviceAllocator : IDeviceAllocator
    {
        private readonly CudaContext context;
        private readonly CudaMemoryPool pool;
        private readonly CudaStream stream;

        public CudaMemoryPoolDeviceAllocator(CudaContext cudaContext)
        {
            context = cudaContext;
            stream = new CudaStream();
            pool = new CudaMemoryPool(cudaContext.Device, true);
        }

        public void Dispose()
        {
        }


        public IDeviceMemory Allocate(long byteCount)
        {
            try
            {
                CUdeviceptr buffer = pool.MemAllocFromPoolAsync(byteCount, stream.Stream);
                return new CudaMemoryPoolDeviceMemory(buffer, () => context.FreeMemoryAsync(buffer, stream.Stream));
            }
            catch (Exception ex)
            {
                GC.Collect(); // Collect unused tensor objects and free GPU memory
                throw new OutOfMemoryException($"Out of GPU memory.");
            }
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }

    public class CudaMemoryPoolDeviceMemory : IDeviceMemory
    {
        private readonly CUdeviceptr pointer;
        private readonly Action freeHandler;

        public CUdeviceptr Pointer => pointer;


        public CudaMemoryPoolDeviceMemory(CUdeviceptr pointer, Action freeHandler)
        {
            this.pointer = pointer;
            this.freeHandler = freeHandler;
        }

        public void Free()
        {
            freeHandler();
        }
    }
}
