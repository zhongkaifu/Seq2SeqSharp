using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp.Optimizer
{

    public class RMSPropOptimizer : IOptimizer
    {
        public static float m_decayRate = 0.999f;
        private static readonly float m_smoothEps = 1e-9f;
        private readonly ConcurrentDictionary<string, Tensor> m_cacheName2V;
        private readonly float m_clipval;

        public RMSPropOptimizer(float clipval, float decayRate = 0.999f)
        {
            Logger.WriteLine($"Creating RMSProp optimizer. GradClip = '{clipval}', LR decay rate = '{decayRate}'");

            m_cacheName2V = new ConcurrentDictionary<string, Tensor>();

            m_clipval = clipval;
            m_decayRate = decayRate;
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
                    if (item != name2tensor[item.Name])
                    {
                        throw new ArgumentException($"Found duplicated weights '{item.Name}'.");
                    }
                    continue;
                }
                name2tensor.Add(item.Name, item);

                if (id2Models.ContainsKey(item.DeviceId) == false)
                {
                    id2Models.Add(item.DeviceId, new List<IWeightTensor>());
                }
                id2Models[item.DeviceId].Add(item);

                if (m_cacheName2V.ContainsKey(item.Name) == false)
                {
                    m_cacheName2V[item.Name] = new Tensor(item.Allocator, DType.Float32, item.Sizes);
                    Ops.Fill(m_cacheName2V[item.Name], 0.0f);

                    Logger.WriteLine($"Added weight '{item.Name}' to optimizer.");
                }
            }

            Parallel.ForEach(id2Models, kv =>
            {
                foreach (IWeightTensor item in kv.Value)
                {
                    WeightTensor m = item as WeightTensor;
                    UpdateWeightsTensor(m, batchSize, step_size, regc, iter);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float regc, int iter)
        {
            try
            {
                Ops.RMSProp(m.TWeight, m.TGradient, m_cacheName2V[m.Name], batchSize, step_size, m_clipval, regc, m_decayRate, m_smoothEps);
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
