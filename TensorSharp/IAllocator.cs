namespace TensorSharp
{
    public enum BlasEnum
    {
        DotNet,
        MKL,
        CUDA
    }


    public interface IAllocator
    {
        BlasEnum BlasEnum { get; }
        int DeviceId { get; }
        Storage Allocate(DType elementType, long elementCount);

        float GetAllocatedMemoryRatio();
    }
}
