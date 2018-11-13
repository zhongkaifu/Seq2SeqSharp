using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class WeightMatrixList
    {
        public List<WeightMatrix> WeightMatrixs = new List<WeightMatrix>();
        public int index = 0;

    }

    public class WeightMatrixFactory
    {
        //private object locker = new object();
        ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixList>> buffer = new ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixList>>();
        public WeightMatrix CreateWeightMatrix(int row, int column)
        {
            var k = buffer.GetOrAdd(row, x => new ConcurrentDictionary<int, WeightMatrixList>());
            var mList = k.GetOrAdd(column, x => new WeightMatrixList());

            bool newMatrix = false;
            WeightMatrix r;
            if (mList.index == mList.WeightMatrixs.Count)
            {
                r = new WeightMatrix(row, column);
                mList.WeightMatrixs.Add(r);
                newMatrix = true;
            }
            else
            {
                r = mList.WeightMatrixs[mList.index];
            }

            mList.index++;

            if (newMatrix == false)
            {
                r.ClearGradient();
            }

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
    }

     
    [Serializable]
    public class WeightMatrix : IWeightMatrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; } 
        public float[] Weight { get; set; }
        public float[] Gradient { get; set; }
        public float[] Cash { get; set; }
        public float[] LrW { get; set; }
        public HashSet<int> RowToBeUpdated { get; set; } = new HashSet<int>();

        //DEBUG variable
        public float AvgLearningRate { get; set; }
        //DEBUG variable

        public WeightMatrix( )
        {
          
        }

        public float[] ToWeightArray()
        {
            return Weight;
        }

        public void SetWeightArray(float[] v)
        {
            Weight = v;
        }

        public void SetGradientFromArray(float[] array)
        {
            Gradient = array;
        }

        public void ClearGradient()
        {
            Array.Clear(Gradient, 0, Gradient.Length);
        }

        public void ClearWeight()
        {
            Array.Clear(Weight, 0, Weight.Length);
        }

        public WeightMatrix(int rows, int columns,  bool normal=false)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
            this.Cash = new float[n];
            this.LrW = new float[n];

            var scale = (float)Math.Sqrt(1.0 / (rows * columns ));
            if (normal)
            {
                scale = 0.08f;
            }
            for (int i = 0; i < n; i++)
            {
                this.Weight[i] = RandomGenerator.NormalRandom(0.0f, scale);  
            }

        }

        public WeightMatrix(int rows, int columns)
        {
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;
            this.Weight = new float[n];
            this.Gradient = new float[n];
        }

        public WeightMatrix(int rows, int columns, float c)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
            this.Cash = new float[n];
            this.LrW = new float[n];

            if (c != 0.0)
            {
                for (int i = 0; i < n; i++)
                {
                    this.Weight[i] = c;
                }
            }        
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            var offset = this.Columns * row;
            Array.Copy(val, 0, Weight, offset, val.Length);
        }

        public WeightMatrix Clone()
        {
            var v= new WeightMatrix(this.Rows, this.Columns, 0);
            var n = this.Weight.Length;
            for (int i = 0; i < n; i++)
            {
                v.Weight[i] = this.Weight[i];
            }
            return v;
        }

        public void CleanCash()
        {
            Cash = new float[Cash.Length];
            LrW = new float[LrW.Length];
        }


        public float GetWeightAt(int offset)
        {
            return Weight[offset];
        }

        public void SetGradientAt(float val, int offset)
        {
            Gradient[offset] = val;
        }

        public void SetWeightAt(float val, int offset)
        {
            Weight[offset] = val;
        }

        public void SetGradientByWeight(IWeightMatrix src)
        {
            WeightMatrix m = src as WeightMatrix;
            Gradient = m.Weight;
        }
    }
}
