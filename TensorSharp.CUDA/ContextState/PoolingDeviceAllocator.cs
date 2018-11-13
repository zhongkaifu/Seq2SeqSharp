using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ManagedCuda.BasicTypes;
using ManagedCuda;

namespace TensorSharp.CUDA.ContextState
{
    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        private const long MemoryAlignment = 256;

        private readonly CudaContext context;
        private Dictionary<long, Queue<IDeviceMemory>> pools = new Dictionary<long, Queue<IDeviceMemory>>();
        private static object locker = new object();

        public PoolingDeviceAllocator(CudaContext context)
        {
            this.context = context;
        }

        public IDeviceMemory Allocate(long byteCount)
        {
            var size = PadToAlignment(byteCount, MemoryAlignment);


            Queue<IDeviceMemory> sizedPool;

            lock (locker)
            {
                if (pools.TryGetValue(size, out sizedPool))
                {
                    if (sizedPool.Count > 0)
                    {
                        var result = sizedPool.Dequeue();

                        // HACK  bizarrely, Queue.Dequeue appears to sometimes return null, even when there are many elements in the queue,
                        // and when the queue is only ever accessed from one thread.
                        if (result != null)
                            return result;
                    }
                }
                else
                {
                    sizedPool = new Queue<IDeviceMemory>();
                    pools.Add(size, sizedPool);
                }


                // If control flow gets to this point, sizedPool exists in the dictionary and is empty.

                var buffer = context.AllocateMemory(size);
                BasicDeviceMemory devMemory = null;
                devMemory = new BasicDeviceMemory(buffer, () =>
                {
                    lock (locker)
                    {
                        sizedPool.Enqueue(devMemory);
                    }
                });

                return devMemory;
            }
        }

        public void Dispose()
        {
            lock (locker)
            {
                foreach (var kvp in pools)
                {
                    foreach (var item in kvp.Value)
                    {
                        item.Free();
                    }
                }

                pools.Clear();
            }
        }

        private static long PadToAlignment(long size, long alignment)
        {
            return ((size + alignment - 1) / alignment) * alignment;
        }
    }
}
