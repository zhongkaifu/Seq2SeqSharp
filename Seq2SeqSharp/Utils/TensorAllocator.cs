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
using System;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.MatrixMul;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Utils
{
    public static class TensorAllocator
    {
        private static IAllocator[] m_allocator = null;
        private static TSCudaContext m_cudaContext = null;
        private static int[] m_deviceIds;
        private static ProcessorTypeEnums m_archType;

        public static void InitDevices(ProcessorTypeEnums archType, int[] ids, float memoryUsageRatio = 0.9f, string[] compilerOptions = null, string mklInstructions = "AVX2", bool enableTensorCore = true, CudaMemoryDeviceAllocatorType allocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool, DType elementType = DType.Float32)
        {
            if (m_allocator != null)
            {
                // The tensor allocator has already been initialized
                return;
            }

            m_archType = archType;
            m_deviceIds = ids;
            m_allocator = new IAllocator[m_deviceIds.Length];
            CudaMatrixMulMM.EnableTensorCore = enableTensorCore;

            if (m_archType == ProcessorTypeEnums.GPU)
            {
                m_cudaContext = new TSCudaContext(m_deviceIds, memoryUsageRatio, compilerOptions, allocatorType, elementType);
                m_cudaContext.Precompile();
                m_cudaContext.CleanUnusedPTX();

                foreach (int deviceId in m_deviceIds)
                {
                    Logger.WriteLine($"Initialize CUDA device '{deviceId}'");
                    int idx = GetDeviceIdIndex(deviceId);
                    m_allocator[idx] = new CudaAllocator(m_cudaContext, deviceId);
                }

            }
            else
            {
                foreach (int deviceId in m_deviceIds)
                {
                    int idx = GetDeviceIdIndex(deviceId);
                    m_allocator[idx] = new CpuAllocator((archType == ProcessorTypeEnums.CPU_MKL) ? BlasEnum.MKL : BlasEnum.DotNet, mklInstructions: mklInstructions);
                }
            }
        }

        public static IAllocator Allocator(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            return m_allocator[idx];
        }

        public static int GetDeviceIdIndex(int id)
        {
            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                if (m_deviceIds[i] == id)
                {
                    return i;
                }
            }

            string strIds = string.Empty;
            foreach (var item in m_deviceIds)
            {
                strIds = strIds + " " + item.ToString();
            }

            throw new ArgumentException($"Failed to get deviceId '{id}', deviceId List = {strIds}");
        }
    }
}
