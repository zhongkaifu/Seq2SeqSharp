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

        private readonly CudaContext m_context;
        private readonly object locker = new object();

        private readonly ulong m_ulAvailMemByteInTotal;
        private CUdeviceptr m_memPoolPtr;
        private readonly SizeT m_startMemAddr;
        private readonly SizeT m_endMemAddr;

        private readonly SortedDictionary<ulong, ulong> m_usedAddr2Size;

        public PoolingDeviceAllocator(CudaContext context, float memoryUsageRatio = 0.9f)
        {
            m_context = context;
            context.SetCurrent();

            m_ulAvailMemByteInTotal = (ulong)((ulong)context.GetFreeDeviceMemorySize() * memoryUsageRatio);

            m_memPoolPtr = context.AllocateMemory(m_ulAvailMemByteInTotal);

            m_startMemAddr = m_memPoolPtr.Pointer;
            m_endMemAddr = m_startMemAddr + m_ulAvailMemByteInTotal;

            m_usedAddr2Size = new SortedDictionary<ulong, ulong>();

            Logger.WriteLine($"Allocated Cuda memory: {m_ulAvailMemByteInTotal}, address from '{m_startMemAddr}' to '{m_endMemAddr}'");
        }

        public float GetAllocatedMemoryRatio()
        {
            lock (locker)
            {
                ulong allocatedMemByte = 0;
                foreach (var pair in m_usedAddr2Size)
                {
                    allocatedMemByte += pair.Value;
                }

                return (float)((float)allocatedMemByte / (float)m_ulAvailMemByteInTotal);
            }
        }

        private CUdeviceptr AllocateMemory(ulong size)
        {
            lock (locker)
            {
                SizeT currMemAddr = m_startMemAddr;
                SizeT currMemAddrEnd;

                foreach (var kv in m_usedAddr2Size)
                {
                    currMemAddrEnd = currMemAddr + size;

                    if (currMemAddrEnd > m_endMemAddr)
                    {
                        GC.Collect(); // Collect unused tensor objects and free GPU memory
                        throw new OutOfMemoryException($"Out of GPU memory. Current memory usage = '{GetAllocatedMemoryRatio() * 100.0f:F}%'");
                    }

                    if (currMemAddrEnd < kv.Key)
                    {
                        m_usedAddr2Size.Add(currMemAddr, size);
                        return new CUdeviceptr(currMemAddr);
                    }
                    else
                    {
                        currMemAddr = kv.Key + kv.Value;
                    }
                }

                currMemAddrEnd = currMemAddr + size;
                if (currMemAddrEnd > m_endMemAddr)
                {
                    GC.Collect(); // Collect unused tensor objects and free GPU memory
                    throw new OutOfMemoryException($"Out of GPU memory. Current memory usage = '{GetAllocatedMemoryRatio() * 100.0f:F}%'");
                }

                m_usedAddr2Size.Add(currMemAddr, size);
                return new CUdeviceptr(currMemAddr);
            }
        }

        public IDeviceMemory Allocate(long byteCount)
        {
            ulong size = PadToAlignment(byteCount, MemoryAlignment);

            lock (locker)
            {            
                CUdeviceptr buffer = AllocateMemory(size);

                BasicDeviceMemory devMemory = null;
                devMemory = new BasicDeviceMemory(buffer, () =>
                {
                    lock (locker)
                    {
                        m_usedAddr2Size.Remove(devMemory.Pointer.Pointer);
                    }
                });

                return devMemory;
            }
        }

        public void Dispose()
        {           
            m_context.SetCurrent();
            m_context.FreeMemory(m_memPoolPtr);
        }

        private static ulong PadToAlignment(long size, long alignment)
        {
            return (ulong)(((size + alignment - 1) / alignment) * alignment);
        }
    }
}
