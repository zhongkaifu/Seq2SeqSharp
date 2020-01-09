namespace TensorSharp.Cpu
{
    public class CpuAllocator : IAllocator
    {
        public CpuAllocator()
        {
        }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CpuStorage(this, elementType, elementCount);
        }
    }
}
