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
    public class WeightTensor : IWeightMatrix,  IDisposable
    {
        public Tensor TWeight;
        public Tensor TGradient;
        public Tensor TLrW;
        public Tensor TCash;

        public int Rows { get; set; }
        public int Columns { get; set; }

        public Dictionary<int, int> RowToBeUpdated { get; set; } = new Dictionary<int, int>();

        public int DeviceId { get; set; }


        public WeightTensor(int rows, int columns, int deviceId, bool normal = false)
        {
            DeviceId = deviceId;
            Rows = rows;
            Columns = columns;
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

            var allocator = TensorAllocator.Allocator(deviceId);

            TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TCash = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TCash, 0.0f);

            TLrW = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TLrW, 0.0f);

            TWeight = Tensor.FromArray(allocator, weight).View(Rows, Columns);
        }

        public WeightTensor(int rows, int columns, int deviceId)
        {
            DeviceId = deviceId;
            Rows = rows;
            Columns = columns;

            var allocator = TensorAllocator.Allocator(deviceId);

            TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            TWeight = new Tensor(allocator, DType.Float32, Rows, Columns);

            Ops.Fill(TGradient, 0.0f);
        }


        public WeightTensor(int rows, int columns, Tensor weight, int deviceId, bool graident = true)
        {
            DeviceId = deviceId;
            Rows = rows;
            Columns = columns;

            TWeight = weight;

            if (graident)
            {
                var allocator = TensorAllocator.Allocator(deviceId);

                TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
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


        public WeightTensor(int rows, int columns, float c, int deviceId)
        {
            DeviceId = deviceId;
            Rows = rows;
            Columns = columns;

            var n = rows * columns;

            var allocator = TensorAllocator.Allocator(deviceId);

            TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TCash = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TCash, 0.0f);

            TLrW = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TLrW, 0.0f);

            TWeight = new Tensor(allocator, DType.Float32, Rows, Columns);
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

            //  Ops.Copy(TGradient, m.TWeight);

            TGradient.Dispose();
            TGradient = m.TWeight;

            m.TWeight = null;
        }

        public void CopyWeights(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;

            Ops.Copy(TWeight, m.TWeight);
        }

        private object locker = new object();
        public void AddGradient(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;

            Tensor t = new Tensor(TGradient.Allocator, DType.Float32, Rows, Columns);
            Ops.Copy(t, m.TGradient);

            lock (locker)
            {
                Ops.Add(TGradient, TGradient, t);
                foreach (var kv in m.RowToBeUpdated)
                {
                    if (RowToBeUpdated.ContainsKey(kv.Key) == false)
                    {
                        RowToBeUpdated.Add(kv.Key, kv.Value);
                    }
                    else
                    {
                        RowToBeUpdated[kv.Key] += kv.Value;
                    }
                }
            }

            t.Dispose();
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
    }
}
