using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
     
    [Serializable]
    public class WeightMatrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; } 
        public float[] Weight { get; set; }
        public float[] Gradient { get; set; }
        public float[] Cash { get; set; }

        public WeightMatrix( )
        {
          
        }
        public WeightMatrix(float[] weights)
        {
            this.Rows = weights.Length;
            this.Columns = 1; 
          //  this.Weight = new float[this.Rows];
            this.Gradient = new float[this.Rows];
            this.Cash = new float[this.Rows];
              this.Weight = weights ;
             
        }

        public WeightMatrix CloneWithSharedParameters()
        {
            WeightMatrix m = new WeightMatrix();

            m.Rows = Rows;
            m.Columns = Columns;
            m.Weight = Weight;
            m.Gradient = new float[Gradient.Length];
            m.Cash = new float[Cash.Length];
    //        m.Delta = new float[Delta.Length];

            return m;
        }

        public WeightMatrix(int rows, int columns,  bool normal=false)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
            this.Cash = new float[n];
        //    this.Delta = new float[n];

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


        //private static ConcurrentDictionary<int, ConcurrentQueue<float[]>> size2QW = new ConcurrentDictionary<int, ConcurrentQueue<float[]>>();
        //public bool UseArrayPool = true;

        public WeightMatrix(int rows, int columns)
        {
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;
                   this.Weight = new float[n];
            this.Gradient = new float[n];

            //var q = size2QW.GetOrAdd(n, x => new ConcurrentQueue<float[]>());
            //float[] w;
            //if (q.TryDequeue(out w))
            //{
            //    this.Weight = w;
            //}
            //else
            //{
            //    this.Weight = new float[n];
            //}
        }

        //~WeightMatrix()
        //{
        //    if (UseArrayPool && size2QW.Count < 10000)
        //    {
        //        var q = size2QW.GetOrAdd(Weight.Length, x => new ConcurrentQueue<float[]>());
        //        q.Enqueue(this.Weight);
        //    }
        //}

        public WeightMatrix(int rows, int columns, float c)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
       //     this.Delta = new float[n];
            this.Cash = new float[n];

            if (c != 0.0)
            {
                for (int i = 0; i < n; i++)
                {
                    this.Weight[i] = c;
                }
            }        
        }

        public override string ToString()
        {
            
            return "{"+Rows.ToString()+","+Columns.ToString()+"}";
        }
        public float Get(int x, int y)
        {
            var ix = ((this.Columns * x) + y)  ;
            return this.Weight[ix];
        }

        public void Set(int x, int y, float v)
        {
            var ix = ((this.Columns * x) + y)  ;
              this.Weight[ix]=v;
        }

        public void Add(int x, int y, float v)
        {
            var ix = ((this.Columns * x) + y)  ;
            this.Weight[ix] += v;
        }

        public float Get_Grad(int x, int y )
        {
            var ix = ((this.Columns * x) + y)  ;
            return this.Gradient[ix];
        }

        public void Set_Grad(int x, int y,   float v)
        {
            var ix = ((this.Columns * x) + y)  ;
            this.Gradient[ix] = v;
        }

        public void Add_Grad(int x, int y,  float v)
        {
            var ix = ((this.Columns * x) + y)  ;
            this.Gradient[ix] += v;
        }

        public WeightMatrix CloneAndZero()
        {
            return new WeightMatrix(this.Rows, this.Columns, 0);

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
 
    }




}
