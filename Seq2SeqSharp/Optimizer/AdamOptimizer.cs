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

    public class AdamOptimizer
    {
        static float m_beta1 = 0.9f;
        static float m_beta2 = 0.98f;
        static float m_smoothEps = 1e-9f;

        ConcurrentDictionary<string, Tensor> m_cacheName2V;
        ConcurrentDictionary<string, Tensor> m_cacheName2M;
        float m_clipval;

        public AdamOptimizer(float clipval, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            Logger.WriteLine($"Creating Adam optimizer. GradClip = '{clipval}', Beta1 = '{beta1}', Beta2 = '{beta2}'");

            m_cacheName2V = new ConcurrentDictionary<string, Tensor>();
            m_cacheName2M = new ConcurrentDictionary<string, Tensor>();

            m_clipval = clipval;
            m_beta1 = beta1;
            m_beta2 = beta2;
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
            Ops.Adam(m.TWeight, m.TGradient, m_cacheName2V[m.Name], m_cacheName2M[m.Name], batchSize, step_size, clipval, regc, m_beta2, m_beta1, iter, m_smoothEps);
        }

    }
}
