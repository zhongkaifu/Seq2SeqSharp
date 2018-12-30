using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public class WeightMatrixList
    {
        public List<WeightMatrix> WeightMatrixs = new List<WeightMatrix>();
        public int index = 0;

    }

    public class WeightMatrixFactory : IWeightFactory
    {
        //private object locker = new object();
        ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixList>> buffer = new ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixList>>();
        public WeightMatrix CreateWeightMatrix(int row, int column, bool cleanWeights = false)
        {
            var k = buffer.GetOrAdd(row, x => new ConcurrentDictionary<int, WeightMatrixList>());
            var mList = k.GetOrAdd(column, x => new WeightMatrixList());

            WeightMatrix r;
            if (mList.index == mList.WeightMatrixs.Count)
            {
                r = new WeightMatrix(row, column);
                if (cleanWeights)
                {
                    r.ClearWeight();
                }

                mList.WeightMatrixs.Add(r);
            }
            else
            {
                r = mList.WeightMatrixs[mList.index];
                r.ClearGradient();
            }

            mList.index++;


            return r;

        }

        public void Clean()
        {
            foreach (var kv in buffer)
            {
                foreach (var subKV in kv.Value)
                {
                    subKV.Value.index = 0;
                }
            }

        }

        public IWeightMatrix CreateWeights(int row, int column)
        {
            return CreateWeightMatrix(row, column);
        }

        public IWeightMatrix CreateWeights(int row, int column, bool cleanWeights)
        {
            return CreateWeightMatrix(row, column, cleanWeights);
        }
    }
}
