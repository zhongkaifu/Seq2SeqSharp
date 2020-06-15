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

        private readonly SizeT m_availMemByteInTotal;
        private CUdeviceptr m_memPoolPtr;
        private readonly SizeT m_startMemAddr;
        private readonly SizeT m_endMemAddr;

        private SortedDictionary<ulong, ulong> m_usedAddr2Size;

        public PoolingDeviceAllocator(CudaContext context, float memoryUsageRatio = 0.9f)
        {
            m_context = context;
            context.SetCurrent();

            long av = context.GetFreeDeviceMemorySize();
            m_availMemByteInTotal = (SizeT)((long)(av * memoryUsageRatio));
            m_memPoolPtr = context.AllocateMemory(m_availMemByteInTotal);

            m_startMemAddr = m_memPoolPtr.Pointer;
            m_endMemAddr = m_startMemAddr + m_availMemByteInTotal;

            m_usedAddr2Size = new SortedDictionary<ulong, ulong>();

            Logger.WriteLine($"Allocated Cuda memory: {m_availMemByteInTotal}, address from '{m_startMemAddr}' to '{m_endMemAddr}'");
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
                        throw new OutOfMemoryException($"Out of GPU memory. currMemAddrEnd = '{currMemAddrEnd}'('{currMemAddr}' + '{size}'), endMemAddr = '{m_endMemAddr}'");
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
                    throw new OutOfMemoryException($"Out of GPU memory. currMemAddrEnd = '{currMemAddrEnd}'('{currMemAddr}' + '{size}'), endMemAddr = '{m_endMemAddr}'");
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
