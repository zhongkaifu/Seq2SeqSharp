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

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
