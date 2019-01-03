using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public class WeightTensorList
    {
        public List<WeightTensor> WeightTensors = new List<WeightTensor>();
        public int index = 0;

    }

    public class WeightTensorFactory : IWeightFactory
    {
        ConcurrentDictionary<int, ConcurrentDictionary<int, WeightTensorList>> buffer = new ConcurrentDictionary<int, ConcurrentDictionary<int, WeightTensorList>>();
        List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor CreateWeightTensor(int row, int column, Tensor w, Tensor g)
        {
            WeightTensor t = new WeightTensor(row, column, w, g);
            weights.Add(t);

            return t;
        }

        public WeightTensor CreateWeightTensor(int row, int column, Tensor w, bool gradient = true)
        {
            WeightTensor t = new WeightTensor(row, column, w, gradient);
            weights.Add(t);

            return t;
        }


        public WeightTensor CreateWeightTensor(int row, int column, bool cleanWeights = false)
        {
            var k = buffer.GetOrAdd(row, x => new ConcurrentDictionary<int, WeightTensorList>());
            var mList = k.GetOrAdd(column, x => new WeightTensorList());

            WeightTensor r;
            if (mList.index == mList.WeightTensors.Count)
            {
                r = new WeightTensor(row, column);
                if (cleanWeights)
                {
                    r.ClearWeight();
                }

                mList.WeightTensors.Add(r);
            }
            else
            {
                r = mList.WeightTensors[mList.index];
                if (cleanWeights)
                {
                    r.ClearWeight();
                }
                r.ClearGradient();
            }

            mList.index++;


            return r;

        }

        public void Clear()
        {
            foreach (var kv in buffer)
            {
                foreach (var subKV in kv.Value)
                {
                    subKV.Value.index = 0;

                    foreach (var item in subKV.Value.WeightTensors)
                    {
                        item.Dispose();
                    }
                }
            }

            buffer.Clear();

            foreach (var item in weights)
            {
                item.Dispose();

            }
            weights.Clear();

        }

        public IWeightMatrix CreateWeights(int row, int column)
        {
            return CreateWeightTensor(row, column);
        }

        public IWeightMatrix CreateWeights(int row, int column, bool cleanWeights)
        {
            return CreateWeightTensor(row, column, cleanWeights);
        }
    }
}
