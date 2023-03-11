using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorSharp.CUDA.ContextState
{
    public enum CudaMemoryDeviceAllocatorType
    {
        Basic,
        CudaMemoryPool,
        CustomMemoryPool
    }
}
