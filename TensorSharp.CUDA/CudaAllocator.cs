using System;

namespace TensorSharp.CUDA
{
    [Serializable]
    public class CudaAllocator : IAllocator
    {
        private readonly TSCudaContext context;
        private readonly int deviceId;

        public CudaAllocator(TSCudaContext context, int deviceId)
        {
            this.context = context;
            this.deviceId = deviceId;
        }

        public TSCudaContext Context => context;
        public int DeviceId => deviceId;

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, context, context.CudaContextForDevice(deviceId), elementType, elementCount);
        }
    }
}
