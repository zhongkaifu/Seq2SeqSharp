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

        public WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false)
        {
            WeightTensor t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);

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

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false)
        {
            WeightTensor r = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);
            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false, string name = "")
        {
            WeightTensor r = new WeightTensor(sizes, deviceId, name);
            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public void Dispose()
        {
            foreach (var item in weights)
            {
                item.Dispose();
            }
            weights.Clear();
        }
    }
}
