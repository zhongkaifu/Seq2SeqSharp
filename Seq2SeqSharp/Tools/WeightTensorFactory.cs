using System;
using System.Collections.Generic;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        private readonly List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor BuildPositionWeightTensor(int row, int column, int deviceId, string name = "", bool isTrainable = false)
        {
            WeightTensor t = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable);

          //  double numTimescales = (float)column / 2;
            double logTimescaleIncrement = Math.Log(10000.0f) / (float)(column);
            float[] posWeights = new float[row * column];

            double sqrtC = Math.Sqrt(column);
            for (int p = 0; p < row; ++p)
            {
                for (int i = 0; i < column; i+=2)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));
                    posWeights[p * column + i] = (float)(Math.Sin(v) / sqrtC);
                    posWeights[p * column + i + 1] = (float)(Math.Cos(v) / sqrtC);
                }
            }

            t.TWeight.CopyFrom(posWeights);

            weights.Add(t);

            return t;
        }

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false, string name = "", bool isTrainable = false, IComputeGraph graphToBind = null)
        {
            WeightTensor r = new WeightTensor(new long[2] { row, column }, deviceId, name: name, isTrainable: isTrainable, graphToBind: graphToBind);

            if (cleanWeights)
            {
                r.CleanWeight();
            }

            weights.Add(r);

            return r;
        }

        public WeightTensor CreateWeightTensor(long[] sizes, int deviceId, bool cleanWeights = false, string name = "", IComputeGraph graphToBind = null)
        {
            WeightTensor r = new WeightTensor(sizes, deviceId, name, graphToBind: graphToBind);

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
