// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;

namespace TensorSharp.CUDA.ContextState
{
    public class MemAddrPair
    {
        public SizeT startMemAddr;
        public SizeT endMemAddr;

        public MemAddrPair(SizeT startMemAddr, SizeT endMemAddr)
        {
            this.startMemAddr = startMemAddr;
            this.endMemAddr = endMemAddr;
        }
    }

    public class PoolingDeviceAllocator : IDeviceAllocator
    {
        private const long MemoryAlignment = 256;

        private readonly CudaContext m_context;
        private readonly object locker = new object();

        private ulong m_ulAvailMemByteInTotal;
        private ulong m_ulReservedMemByte;

        private List<CUdeviceptr> m_memPoolPtrs = new List<CUdeviceptr>();
        private List<MemAddrPair> m_memAddrs = new List<MemAddrPair>();
        private readonly List<SortedDictionary<ulong, ulong>> m_usedAddr2Sizes = new List<SortedDictionary<ulong, ulong>>();

        public PoolingDeviceAllocator(CudaContext context, float memoryUsageRatio = 0.9f)
        {
            m_context = context;
            context.SetCurrent();

            ulong ulFreeMemoryByte = (ulong)context.GetFreeDeviceMemorySize();
            m_ulAvailMemByteInTotal = (ulong)(ulFreeMemoryByte * memoryUsageRatio);
            m_ulReservedMemByte = ulFreeMemoryByte - m_ulAvailMemByteInTotal;

            CUdeviceptr memPoolPtr = context.AllocateMemory(m_ulAvailMemByteInTotal);
            m_memPoolPtrs.Add(memPoolPtr);

            SizeT m_startMemAddr = memPoolPtr.Pointer;
            SizeT m_endMemAddr = m_startMemAddr + m_ulAvailMemByteInTotal;
            m_memAddrs.Add(new MemAddrPair(m_startMemAddr, m_endMemAddr));
            m_usedAddr2Sizes.Add(new SortedDictionary<ulong, ulong>());

            Logger.WriteLine($"Allocated Cuda memory: {m_ulAvailMemByteInTotal}, address from '{m_startMemAddr}' to '{m_endMemAddr}'");
        }

        public float GetAllocatedMemoryRatio()
        {
            lock (locker)
            {
                ulong allocatedMemByte = 0;
                foreach (var item in m_usedAddr2Sizes)
                {
                    foreach (var pair in item)
                    {
                        allocatedMemByte += pair.Value;
                    }
                }

                return (float)((float)allocatedMemByte / (float)m_ulAvailMemByteInTotal);
            }
        }

        private CUdeviceptr AllocateMemory(ulong size)
        {
            lock (locker)
            {
                for (int i = 0; i < m_memAddrs.Count; i++)
                {
                    MemAddrPair memAddrPair = m_memAddrs[i];
                    SizeT currMemAddr = memAddrPair.startMemAddr;
                    SizeT currMemAddrEnd;

                    foreach (var kv in m_usedAddr2Sizes[i])
                    {
                        currMemAddrEnd = currMemAddr + size;
                        if (currMemAddrEnd < kv.Key)
                        {
                            m_usedAddr2Sizes[i].Add(currMemAddr, size);
                            return new CUdeviceptr(currMemAddr);
                        }
                        else
                        {
                            currMemAddr = kv.Key + kv.Value;
                        }
                    }

                    currMemAddrEnd = currMemAddr + size;
                    if (currMemAddrEnd < memAddrPair.endMemAddr)
                    {
                        m_usedAddr2Sizes[i].Add(currMemAddr, size);
                        return new CUdeviceptr(currMemAddr);

                    }

                    if (i == m_memAddrs.Count - 1)
                    {
                        //We did not find any existing pool has enough available memory, so let's create a new pool and try to allocate memory from it. 
                        CreateNewMemPool(size);
                    }
                }

                return new CUdeviceptr();
            }
        }

        private void CreateNewMemPool(ulong size)
        {
            GC.Collect(); // Collect unused tensor objects and free GPU memory

            m_context.SetCurrent();
            long lAvailMemByte = (long)m_context.GetFreeDeviceMemorySize() - (long)m_ulReservedMemByte;
            if (lAvailMemByte < 0 || (long)size > lAvailMemByte)
            {
                throw new OutOfMemoryException($"Out of GPU memory. Current memory usage = '{GetAllocatedMemoryRatio() * 100.0f:F}%'");
            }

            Logger.WriteLine($"Current memory pool does not have enough free memory to allocate '{size}' memory, let's create a new pool. Size = '{lAvailMemByte}'");
            m_ulAvailMemByteInTotal += (ulong)lAvailMemByte;
            CUdeviceptr memPoolPtr = m_context.AllocateMemory(lAvailMemByte);
            m_memPoolPtrs.Add(memPoolPtr);

            SizeT startMemAddr = memPoolPtr.Pointer;
            SizeT endMemAddr = startMemAddr + lAvailMemByte;
            m_memAddrs.Add(new MemAddrPair(startMemAddr, endMemAddr));
            m_usedAddr2Sizes.Add(new SortedDictionary<ulong, ulong>());
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
                        foreach (var item in m_usedAddr2Sizes)
                        {
                            if (item.ContainsKey(devMemory.Pointer.Pointer))
                            {
                                item.Remove(devMemory.Pointer.Pointer);
                            }
                        }
                    }
                });

                return devMemory;
            }
        }

        public void Dispose()
        {           
            m_context.SetCurrent();

            foreach (var item in m_memPoolPtrs) 
            {
                m_context.FreeMemory(item);
            }
        }

        private static ulong PadToAlignment(long size, long alignment)
        {
            return (ulong)(((size + alignment - 1) / alignment) * alignment);
        }
    }
}
