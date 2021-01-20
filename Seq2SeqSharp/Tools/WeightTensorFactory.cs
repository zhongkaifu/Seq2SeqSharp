using System;
using System.Collections.Generic;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        private readonly List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, IComputeGraph graphToBind = null, NormType normType = NormType.None)
        {
            WeightTensor r = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable, normType: normType, graphToBind: graphToBind);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false, string name = "", IComputeGraph graphToBind = null, NormType normType = NormType.None)
        {
            WeightTensor r = new WeightTensor(sizes, deviceId, name, normType: normType, graphToBind: graphToBind);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public void Dispose()
        {
            foreach (WeightTensor item in weights)
            {
                item.Dispose();
            }
            weights.Clear();
        }
    }
}
