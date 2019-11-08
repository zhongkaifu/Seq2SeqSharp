using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{

    public class Optimizer
    {
        static float decay_rate_m = 0.9f;
        static float decay_rate_v = 0.999f;
        static float smooth_eps = 1e-10f;

        ConcurrentDictionary<string, Tensor> m_cacheName2V;
        ConcurrentDictionary<string, Tensor> m_cacheName2M;
        float m_clipval;

        public Optimizer(float clipval)
        {
            Logger.WriteLine($"Creating Adam optimizer. GradClip = '{clipval}'");

            m_cacheName2V = new ConcurrentDictionary<string, Tensor>();
            m_cacheName2M = new ConcurrentDictionary<string, Tensor>();

            m_clipval = clipval;
        }

        public void Clear()
        {
            foreach (var pair in m_cacheName2V)
            {
                Ops.Fill(pair.Value, 0.0f);
            }

            foreach (var pair in m_cacheName2M)
            {
                Ops.Fill(pair.Value, 0.0f);
            }
        }

        public void UpdateWeights(List<IWeightTensor> model, int batchSize, float step_size, float regc, int iter)
        {
            Dictionary<int, List<IWeightTensor>> id2Models = new Dictionary<int, List<IWeightTensor>>();
            HashSet<string> setWeightsName = new HashSet<string>();
            foreach (var item in model)
            {
                if (setWeightsName.Contains(item.Name))
                {
                    throw new ArgumentException($"Found duplicated weights name '{item.Name}'");
                }
                setWeightsName.Add(item.Name);

                if (id2Models.ContainsKey(item.DeviceId) == false)
                {
                    id2Models.Add(item.DeviceId, new List<IWeightTensor>());
                }
                id2Models[item.DeviceId].Add(item);

                if (m_cacheName2V.ContainsKey(item.Name) == false)
                {
                    var allocator = TensorAllocator.Allocator(item.DeviceId);
                    m_cacheName2V[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2V[item.Name], 0.0f);

                    m_cacheName2M[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2M[item.Name], 0.0f);

                    Logger.WriteLine($"Added weight '{item.Name}' to optimizer.");
                }
            }

            Parallel.ForEach(id2Models, kv => 
            {
                foreach (var item in kv.Value)
                {
                    var m = item as WeightTensor;
                    UpdateWeightsTensor(m, batchSize, step_size, m_clipval, regc, iter);
                }
            });
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc, int iter)
        {
            // Ops.RMSProp(m.TWeight, m.TGradient, m.TV, batchSize, step_size, clipval, regc, decay_rate, smooth_eps);
            Ops.Adam(m.TWeight, m.TGradient, m_cacheName2V[m.Name], m_cacheName2M[m.Name], batchSize, step_size, clipval, regc, decay_rate_v, decay_rate_m, iter, smooth_eps);
        }

    }
}
