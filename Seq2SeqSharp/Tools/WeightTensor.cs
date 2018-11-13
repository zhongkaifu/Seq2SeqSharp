using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Security.Permissions;
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

    public class WeightTensorFactory
    {
        //private object locker = new object();
        ConcurrentDictionary<int, ConcurrentDictionary<int, WeightTensorList>> buffer = new ConcurrentDictionary<int, ConcurrentDictionary<int, WeightTensorList>>();
        public WeightTensor CreateWeightTensor(int row, int column)
        {
            var k = buffer.GetOrAdd(row, x => new ConcurrentDictionary<int, WeightTensorList>());
            var mList = k.GetOrAdd(column, x => new WeightTensorList());

            bool newTensor = false;
            WeightTensor r;
            if (mList.index == mList.WeightTensors.Count)
            {
                r = new WeightTensor(row, column);
                mList.WeightTensors.Add(r);
                newTensor = true;
            }
            else
            {
                r = mList.WeightTensors[mList.index];
            }

            mList.index++;

            if (newTensor == false)
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
    public class WeightTensor : IWeightMatrix, ISerializable
    {
        public Tensor TWeight;
        public Tensor TGradient;
        public Tensor TLrW;
        public Tensor TCash;

        public int Rows { get; set; }
        public int Columns { get; set; }

        public HashSet<int> RowToBeUpdated { get; set; } = new HashSet<int>();

        //DEBUG variable
        public float AvgLearningRate { get; set; }
        //DEBUG variable

        public WeightTensor(int rows, int columns, bool normal = false)
        {
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;

            float[] weight = new float[n];


            var scale = (float)Math.Sqrt(1.0 / (rows * columns));
            if (normal)
            {
                scale = 0.08f;
            }
            for (int i = 0; i < n; i++)
            {
                weight[i] = RandomGenerator.NormalRandom(0.0f, scale);
            }

            TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TCash = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TCash, 0.0f);

            TLrW = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TLrW, 0.0f);

            TWeight = Tensor.FromArray(TensorAllocator.Allocator, weight).View(Rows, Columns);
        }

        public WeightTensor(int rows, int columns)
        {
            this.Rows = rows;
            this.Columns = columns;

            TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            TWeight = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);

            Ops.Fill(TGradient, 0.0f);
        }


        public WeightTensor(int rows, int columns, Tensor weight)
        {
            this.Rows = rows;
            this.Columns = columns;

            TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            TWeight = weight;

            Ops.Fill(TGradient, 0.0f);
        }

        public WeightTensor(int rows, int columns, Tensor weight, Tensor gradient)
        {
            this.Rows = rows;
            this.Columns = columns;

            TGradient = gradient;
            TWeight = weight;
        }


        public WeightTensor(int rows, int columns, float c)
        {
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;

            TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TCash = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TCash, 0.0f);

            TLrW = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TLrW, 0.0f);

            TWeight = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TWeight, c);
        }

        public void CleanCash()
        {
            Ops.Fill(TCash, 0.0f);
            Ops.Fill(TLrW, 0.0f);
        }

        public void ClearGradient()
        {
            Ops.Fill(TGradient, 0.0f);
        }

        public void ClearWeight()
        {
            Ops.Fill(TWeight, 0.0f);
        }

        public float GetWeightAt(int offset)
        {
            return TWeight.GetElementAsFloat(0, offset);
        }


        public float GetGradientAt(int offset)
        {
            return TGradient.GetElementAsFloat(0, offset);
        }

        public void SetGradientAt(float val, int offset)
        {
            TGradient.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAt(float val, int offset)
        {
            TWeight.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            TWeight.SetElementsAsFloat(val, row, 0);
        }


        public void SetGradientByWeight(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;
            TGradient = m.TWeight;
        }

        public float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public void SetWeightArray(float[] v)
        {
            TWeight.SetElementsAsFloat(v);
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("Rows", Rows);
            info.AddValue("Columns", Columns);            
            info.AddValue("Weight", ToWeightArray());

        }


        [SecurityPermissionAttribute(SecurityAction.Demand, SerializationFormatter = true)]
        protected WeightTensor(SerializationInfo info, StreamingContext context)
        {
            Rows = info.GetInt32("Rows");
            Columns = info.GetInt32("Columns");

            TWeight = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            TCash = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
            TLrW = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);

            Ops.Fill(TGradient, 0.0f);
            Ops.Fill(TCash, 0.0f);
            Ops.Fill(TLrW, 0.0f);

            float[] weights = (float[])info.GetValue("Weight", typeof(float[]));
            SetWeightArray(weights);            
        }
    }
}
