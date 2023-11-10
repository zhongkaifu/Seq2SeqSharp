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
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace Seq2SeqSharp.Optimizer
{

    public class AdamOptimizer : IOptimizer
    {
        private static float m_beta1 = 0.9f;
        private static float m_beta2 = 0.98f;
        private static readonly float m_smoothEps = 1e-9f;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2V;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2M;
        private readonly float m_clipval;
        private readonly bool m_saveGPUMemoryMode = false;

        public AdamOptimizer(float clipval, float beta1 = 0.9f, float beta2 = 0.98f, bool saveGPUMemoryMode = false)
        {
            if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                Logger.WriteLine($"Creating Adam optimizer. GradClip = '{clipval}', Beta1 = '{beta1}', Beta2 = '{beta2}', SaveGPUMemoryMode = '{saveGPUMemoryMode}'");

            m_cacheName2V = new ConcurrentDictionary<string, Tensor>();
            m_cacheName2M = new ConcurrentDictionary<string, Tensor>();

            m_clipval = clipval;
            m_beta1 = beta1;
            m_beta2 = beta2;
            m_saveGPUMemoryMode = saveGPUMemoryMode;
        }

        public void UpdateWeights(List<IWeightTensor> model, int batchSize, float step_size, float regc, int iter)
        {
            Dictionary<int, List<IWeightTensor>> id2Models = new Dictionary<int, List<IWeightTensor>>();
            Dictionary<string, IWeightTensor> name2tensor = new Dictionary<string, IWeightTensor>();

            foreach (IWeightTensor item in model)
            {
                if (!item.IsTrainable)
                {
                    continue;
                }

                if (name2tensor.ContainsKey(item.Name))
                {
                    throw new ArgumentException($"Found duplicated weights '{item.Name}'.");
                }
                name2tensor.Add(item.Name, item);

                if (id2Models.ContainsKey(item.DeviceId) == false)
                {
                    id2Models.Add(item.DeviceId, new List<IWeightTensor>());
                }
                id2Models[item.DeviceId].Add(item);

                if (m_cacheName2V.ContainsKey(item.Name) == false)
                {
                    m_cacheName2V[item.Name] = new Tensor(m_saveGPUMemoryMode ? new CpuAllocator(BlasEnum.DotNet) : item.Allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2V[item.Name], 0.0f);

                    m_cacheName2M[item.Name] = new Tensor(m_saveGPUMemoryMode ? new CpuAllocator(BlasEnum.DotNet) : item.Allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2M[item.Name], 0.0f);

                    if (Logger.Verbose != Logger.LogVerbose.None && Logger.Verbose != Logger.LogVerbose.Normal && Logger.Verbose != Logger.LogVerbose.Callback)
                        Logger.WriteLine($"Added weight '{item.Name}' to optimizer. Learning rate factor = '{item.LearningRateFactor}'");
                }
            }

            Parallel.ForEach(id2Models, kv =>
            {
                foreach (IWeightTensor item in kv.Value)
                {
                    WeightTensor m = item as WeightTensor;
                    UpdateWeightsTensor(m, batchSize, step_size * m.LearningRateFactor, regc, iter);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float regc, int iter)
        {
            try
            {
                if ((m.Allocator is CudaAllocator) && m_saveGPUMemoryMode)
                {
                    Tensor t1 = new Tensor(m.Allocator, m_cacheName2V[m.Name].ElementType, m_cacheName2V[m.Name].Sizes);
                    Ops.Copy(t1, m_cacheName2V[m.Name]);

                    Tensor t2 = new Tensor(m.Allocator, m_cacheName2M[m.Name].ElementType, m_cacheName2M[m.Name].Sizes);
                    Ops.Copy(t2, m_cacheName2M[m.Name]);


                    Ops.Adam(m.TWeight, m.TGradient, t1, t2, batchSize, step_size, m_clipval, regc, m_beta2, m_beta1, iter, m_smoothEps);

                    Ops.Copy(m_cacheName2V[m.Name], t1);
                    t1.Dispose();

                    Ops.Copy(m_cacheName2M[m.Name], t2);
                    t2.Dispose();
                }
                else
                {              
                    Ops.Adam(m.TWeight, m.TGradient, m_cacheName2V[m.Name], m_cacheName2M[m.Name], batchSize, step_size, m_clipval, regc, m_beta2, m_beta1, iter, m_smoothEps);
                }

            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception: '{err.Message}'");
                Logger.WriteLine(Logger.Level.err, $"Call stack: '{err.StackTrace}'");

                throw;
            }
        }

    }
}
