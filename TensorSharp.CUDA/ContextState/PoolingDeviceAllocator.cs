using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;

namespace TensorSharp.CUDA.ContextState
{
    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        private const long MemoryAlignment = 256;

        private readonly CudaContext context;
      //  private readonly Dictionary<long, Queue<IDeviceMemory>> pools = new Dictionary<long, Queue<IDeviceMemory>>();
        //private long allocatedSize = 0;
        //private long missingCacheSize = 0;
        //private const long maxSize = (long)(1024L * 1024L * 1024L * 4L);
        private readonly object locker = new object();

        private readonly SizeT availMemByteInTotal;
        private CUdeviceptr memPoolPtr;
        private readonly SizeT startMemAddr;
        private readonly SizeT endMemAddr;
     //   private SizeT currMemAddr;

        private SortedDictionary<ulong, ulong> usedAddr2Size;

        public PoolingDeviceAllocator(CudaContext context, float memoryUsageRatio = 0.9f)
        {
            this.context = context;

            context.SetCurrent();

            long av = context.GetFreeDeviceMemorySize();
            availMemByteInTotal = (SizeT)((long)(av * memoryUsageRatio));

            memPoolPtr = context.AllocateMemory(availMemByteInTotal);

            startMemAddr = memPoolPtr.Pointer;
            endMemAddr = startMemAddr + availMemByteInTotal;

            usedAddr2Size = new SortedDictionary<ulong, ulong>();

            Logger.WriteLine($"Allocated Cuda memory: {availMemByteInTotal}, address from '{startMemAddr}' to '{endMemAddr}'");
        }


        private CUdeviceptr AllocateMemory(ulong size)
        {
            lock (locker)
            {
                SizeT currMemAddr = startMemAddr;
                SizeT currMemAddrEnd;

                foreach (var kv in usedAddr2Size)
                {
                    currMemAddrEnd = currMemAddr + size;

                    if (currMemAddrEnd > endMemAddr)
                    {
                        throw new OutOfMemoryException($"Out of GPU memory. currMemAddrEnd = '{currMemAddrEnd}'('{currMemAddr}' + '{size}'), endMemAddr = '{endMemAddr}'");
                    }

                    if (currMemAddrEnd < kv.Key)
                    {
                        usedAddr2Size.Add(currMemAddr, size);
                        return new CUdeviceptr(currMemAddr);
                    }
                    else
                    {
                        currMemAddr = kv.Key + kv.Value;
                    }
                }

                currMemAddrEnd = currMemAddr + size;
                if (currMemAddrEnd > endMemAddr)
                {
                    throw new OutOfMemoryException($"Out of GPU memory. currMemAddrEnd = '{currMemAddrEnd}'('{currMemAddr}' + '{size}'), endMemAddr = '{endMemAddr}'");
                }

                usedAddr2Size.Add(currMemAddr, size);
                return new CUdeviceptr(currMemAddr);
            }
        }

        public void FreeMemory(bool callGC = false)
        {
            lock (locker)
            {
                if (callGC)
                {
                    GC.Collect();
                    GC.WaitForFullGCComplete();
                }

                //foreach (KeyValuePair<long, Queue<IDeviceMemory>> kv in pools)
                //{
                //    while (kv.Value.Count > 0)
                //    {
                //        IDeviceMemory item = kv.Value.Dequeue();
                //        if (item != null)
                //        {
                //            context.FreeMemory(item.Pointer);
                //        }
                //    }
                //}
            }
        }


        public IDeviceMemory Allocate(long byteCount)
        {
            ulong size = PadToAlignment(byteCount, MemoryAlignment);

            lock (locker)
            {
                ////   allocatedSize += size;
                //if (pools.TryGetValue(size, out Queue<IDeviceMemory> sizedPool))
                //{
                //    if (sizedPool.Count > 0)
                //    {
                //        IDeviceMemory result = sizedPool.Dequeue();

                //        usedAddr2Size.Add(result.Pointer.Pointer, size);

                //        // HACK  bizarrely, Queue.Dequeue appears to sometimes return null, even when there are many elements in the queue,
                //        // and when the queue is only ever accessed from one thread.
                //        if (result != null)
                //        {
                //            return result;
                //        }
                //    }
                //}
                //else
                //{
                //    sizedPool = new Queue<IDeviceMemory>();
                //    pools.Add(size, sizedPool);
                //}

                CUdeviceptr buffer = AllocateMemory(size);
                //try
                //{
                //    try
                //    {
                //        // If control flow gets to this point, sizedPool exists in the dictionary and is empty.
                //        context.SetCurrent();
                //        buffer = context.AllocateMemory(size);
                //    }
                //    catch (ManagedCuda.CudaException)
                //    {
                //        FreeMemory(false);
                //        buffer = context.AllocateMemory(size);
                //    }
                //}
                //catch (ManagedCuda.CudaException)
                //{
                //    FreeMemory(true);
                //    buffer = context.AllocateMemory(size);
                //}

                BasicDeviceMemory devMemory = null;
                devMemory = new BasicDeviceMemory(buffer, () =>
                {
                    lock (locker)
                    {
                        usedAddr2Size.Remove(devMemory.Pointer.Pointer);

                      //  sizedPool.Enqueue(devMemory);
                    }
                });

                return devMemory;
            }
        }

        public void Dispose()
        {
            //lock (locker)
            //{
            //    foreach (KeyValuePair<long, Queue<IDeviceMemory>> kvp in pools)
            //    {
            //        foreach (IDeviceMemory item in kvp.Value)
            //        {
            //            item.Free();
            //        }
            //    }

            //    pools.Clear();
            //}
        }

        private static ulong PadToAlignment(long size, long alignment)
        {
            return (ulong)(((size + alignment - 1) / alignment) * alignment);
        }
    }
}
