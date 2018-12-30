using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    [Serializable]
    public class WeightTensor : IWeightMatrix, ISerializable, IDisposable
    {
        public Tensor TWeight;
        public Tensor TGradient;
        public Tensor TLrW;
        public Tensor TCash;

        public int Rows { get; set; }
        public int Columns { get; set; }

        public Dictionary<int, int> RowToBeUpdated { get; set; } = new Dictionary<int, int>();

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


        public WeightTensor(int rows, int columns, Tensor weight, bool graident = true)
        {
            this.Rows = rows;
            this.Columns = columns;

            TWeight = weight;

            if (graident)
            {
                TGradient = new Tensor(TensorAllocator.Allocator, DType.Float32, Rows, Columns);
                Ops.Fill(TGradient, 0.0f);
            }
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

        //~WeightTensor()
        //{
        //    Dispose();
        //}


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

            // Ops.Copy(TGradient, m.TWeight);

            TGradient.Dispose();
            TGradient = m.TWeight;

            m.TWeight = null;
        }

        public float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public int GetMaxWeightIdx()
        {
            float[] weights = ToWeightArray();
            var maxv = weights[0];
            var maxi = 0;
            for (int i = 1; i < weights.Length; i++)
            {
                if (weights[i] > maxv)
                {
                    maxv = weights[i];
                    maxi = i;
                }
            }

            return maxi;
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

        public void Dispose()
        {
            if (TWeight != null)
            {
                TWeight.Dispose();
                TWeight = null;
            }

            if (TGradient != null)
            {
                TGradient.Dispose();
                TGradient = null;
            }

            if (TCash != null)
            {
                TCash.Dispose();
                TCash = null;
            }

            if (TLrW != null)
            {
                TLrW.Dispose();
                TLrW = null;
            }
        }

        public void ReleaseWeight()
        {
            if (TWeight != null)
            {
                TWeight.Dispose();
                TWeight = null;
            }
        }

        public void Save(Stream stream)
        {
            var floatArray1 = ToWeightArray();

            // create a byte array and copy the floats into it...
            var byteArray = new byte[floatArray1.Length * 4];
            Buffer.BlockCopy(floatArray1, 0, byteArray, 0, byteArray.Length);

            stream.Write(byteArray, 0, byteArray.Length);
        }

        public void Load(Stream stream)
        {
            int size = Rows * Columns;
            var byteArray = new byte[size * 4];
            stream.Read(byteArray, 0, byteArray.Length);

            var floatArray2 = new float[byteArray.Length / 4];
            Buffer.BlockCopy(byteArray, 0, floatArray2, 0, byteArray.Length);

            SetWeightArray(floatArray2);
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
