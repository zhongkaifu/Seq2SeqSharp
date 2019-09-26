using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor CreateWeightTensor(int row, int column, Tensor w, Tensor g)
        {
            WeightTensor t = new WeightTensor(row, column, w, g);
            weights.Add(t);

            return t;
        }

        public WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId)
        {
            WeightTensor t = new WeightTensor(row, column, deviceId);

            double numTimescales = (float)column / 2;
            double logTimescaleIncrement = Math.Log(10000.0f) / (numTimescales - 1.0f);
            float[] posWeights = new float[row * column];

            for (int p = 0; p < row; ++p)
            {
                for (int i = 0; i < numTimescales; ++i)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));
                    posWeights[p * column + i] = (float)Math.Sin(v);
                    posWeights[p * column + (int)numTimescales + i] = (float)Math.Cos(v);
                }
            }

            t.TWeight.CopyFrom(posWeights);

            weights.Add(t);

            return t;
        }

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false)
        {
            WeightTensor r = new WeightTensor(row, column, deviceId);

            if (cleanWeights)
            {
                r.ClearWeight();
            }

            weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false)
        {
            WeightTensor r = new WeightTensor(sizes, deviceId);

            if (cleanWeights)
            {
                r.ClearWeight();
            }

            weights.Add(r);

            return r;
        }

        public void Clear()
        {
            foreach (var item in weights)
            {
                item.Dispose();
            }
            weights.Clear();

        }

        public IWeightTensor CreateWeights(int row, int column, int deviceId)
        {
            return CreateWeightTensor(row, column, deviceId);
        }

        public IWeightTensor CreateWeights(int row, int column, int deviceId, bool cleanWeights)
        {
            return CreateWeightTensor(row, column, deviceId, cleanWeights);
        }
    }
}
