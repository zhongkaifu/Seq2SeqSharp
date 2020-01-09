namespace TensorSharp
{
    public interface IAllocator
    {
        Storage Allocate(DType elementType, long elementCount);
    }
}
