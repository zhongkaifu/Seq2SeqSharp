using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using TensorSharp.CUDA.ContextState;

namespace TensorSharp.CUDA
{
    [Serializable]
    public class CudaStorage : Storage
    {
        private readonly CudaContext context;

        private IDeviceMemory bufferHandle;
        private readonly CUdeviceptr deviceBuffer;


        public CudaStorage(IAllocator allocator, TSCudaContext tsContext, CudaContext context, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            TSContext = tsContext;
            this.context = context;

            bufferHandle = tsContext.AllocatorForDevice(DeviceId).Allocate(ByteLength);
            deviceBuffer = bufferHandle.Pointer;
        }

        public TSCudaContext TSContext { get; private set; }

        protected override void Destroy()
        {
            if (bufferHandle != null)
            {
                bufferHandle.Free();
                bufferHandle = null;
            }
        }

        public override string LocationDescription()
        {
            return "CUDA:" + context.DeviceId;
        }

        public int DeviceId => context.DeviceId;

        public CUdeviceptr DevicePtrAtElement(long index)
        {
            long offset = ElementType.Size() * index;
            return new CUdeviceptr(deviceBuffer.Pointer + offset);
        }

        public override float GetElementAsFloat(long index)
        {
            CUdeviceptr ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { float[] result = new float[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.Float64) { double[] result = new double[1]; context.CopyToHost(result, ptr); return (float)result[0]; }
            else if (ElementType == DType.Int32) { int[] result = new int[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.UInt8) { byte[] result = new byte[1]; context.CopyToHost(result, ptr); return result[0]; }
            else
            {
                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }


        public override float[] GetElementsAsFloat(long index, int length)
        {
            CUdeviceptr ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { float[] result = new float[length]; context.CopyToHost(result, ptr); return result; }
            else
            {
                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override void SetElementAsFloat(long index, float value)
        {
            CUdeviceptr ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { context.CopyToDevice(ptr, value); }
            else if (ElementType == DType.Float64) { context.CopyToDevice(ptr, (double)value); }
            else if (ElementType == DType.Int32) { context.CopyToDevice(ptr, (int)value); }
            else if (ElementType == DType.UInt8) { context.CopyToDevice(ptr, (byte)value); }
            else
            {
                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            CUdeviceptr ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { context.CopyToDevice(ptr, value); }
            else
            {
                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            CUdeviceptr dstPtr = DevicePtrAtElement(storageIndex);
            context.SetCurrent();
            context.CopyToDevice(dstPtr, src, byteCount);
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            CUdeviceptr srcPtr = DevicePtrAtElement(storageIndex);

            // Call this method directly instead of CudaContext.CopyToHost because this method supports a long byteCount
            // CopyToHost only supports uint byteCount.
            CUResult res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dst, srcPtr, byteCount);
            if (res != CUResult.Success)
            {
                throw new CudaException(res);
            }
        }
    }
}
