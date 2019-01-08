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

       // private object locker = new object();

        public WeightTensor CreateWeightTensor(int row, int column, Tensor w, Tensor g)
        {
            WeightTensor t = new WeightTensor(row, column, w, g);

       //     lock (locker)
        //    {
                weights.Add(t);
        //    }

            return t;
        }

        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, Tensor w, bool gradient = true)
        {
            WeightTensor t = new WeightTensor(row, column, w, deviceId, gradient);
         //   lock (locker)
         //   {
                weights.Add(t);
         //   }

            return t;
        }


        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false)
        {

            var k = buffer.GetOrAdd(row, x => new ConcurrentDictionary<int, WeightTensorList>());
            var mList = k.GetOrAdd(column, x => new WeightTensorList());

            WeightTensor r;
         //   lock (locker)
         //   {
          //      if (mList.index == mList.WeightTensors.Count)
         //       {
                    r = new WeightTensor(row, column, deviceId);
                    mList.WeightTensors.Add(r);
                //}
                //else
                //{
                //    r = mList.WeightTensors[mList.index];
                //    r.ClearGradient();
                //}

                //mList.index++;

       //     }

            if (cleanWeights)
            {
                r.ClearWeight();
            }

            return r;


        }

        public void Clear()
        {
        //    lock (locker)
        //    {
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

                if (weights == null)
                {
                    throw new InvalidOperationException($"weights is null.");
                }

                foreach (var item in weights)
                {
                    if (item == null)
                    {
                        throw new InvalidOperationException($"weights' item is null.");
                    }

                    item.Dispose();

                }
                weights.Clear();
          //  }

        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId)
        {
            return CreateWeightTensor(row, column, deviceId);
        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId, bool cleanWeights)
        {
            return CreateWeightTensor(row, column, deviceId, cleanWeights);
        }
    }
}
