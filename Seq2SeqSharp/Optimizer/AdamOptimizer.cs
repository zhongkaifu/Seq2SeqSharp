using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{

    public class AdamOptimizer
    {
        private static float m_beta1 = 0.9f;
        private static float m_beta2 = 0.98f;
        private static readonly float m_smoothEps = 1e-9f;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2V;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2M;
        private readonly float m_clipval;

        public AdamOptimizer(float clipval, float beta1 = 0.9f, float beta2 = 0.98f)
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

            foreach (IWeightTensor item in model)
            {
                if (!item.IsTrainable)
                {
                    continue;
                }

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
                    IAllocator allocator = TensorAllocator.Allocator(item.DeviceId);
                    m_cacheName2V[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2V[item.Name], 0.0f);

                    m_cacheName2M[item.Name] = new Tensor(allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2M[item.Name], 0.0f);

                    Logger.WriteLine($"Added weight '{item.Name}' to optimizer.");
                }
            }

            Parallel.ForEach(id2Models, kv =>
            {
                foreach (IWeightTensor item in kv.Value)
                {
                    WeightTensor m = item as WeightTensor;
                    UpdateWeightsTensor(m, batchSize, step_size, m_clipval, regc, iter);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc, int iter)
        {
            try
            {
                float clip_coef = 1.0f;

                //float normVal = Ops.NormAll(m.TGradient, 2.0f);
                //float clip_coef = 0.5f / (normVal + 1e-6f);
                //if (clip_coef > 1.0f)
                //{
                //    clip_coef = 1.0f;
                //}

                // Ops.RMSProp(m.TWeight, m.TGradient, m.TV, batchSize, step_size, clipval, regc, decay_rate, smooth_eps);
                Ops.Adam(m.TWeight, m.TGradient, m_cacheName2V[m.Name], m_cacheName2M[m.Name], batchSize, step_size, clip_coef, regc, m_beta2, m_beta1, iter, m_smoothEps);
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception: '{err.Message}'");
                Logger.WriteLine(Logger.Level.err, $"Call stack: '{err.StackTrace}'");

                throw err;
            }
        }

    }
}
