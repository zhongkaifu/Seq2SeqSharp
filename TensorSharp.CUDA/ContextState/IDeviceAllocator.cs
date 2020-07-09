using ManagedCuda.BasicTypes;
using System;

namespace TensorSharp.CUDA.ContextState
{
    public interface IDeviceMemory
    {
        CUdeviceptr Pointer { get; }

        void Free();
    }

    public interface IDeviceAllocator : IDisposable
    {
        IDeviceMemory Allocate(long byteCount);
        float GetAllocatedMemoryRatio();
    }
}
