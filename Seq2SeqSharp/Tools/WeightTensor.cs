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
using AdvUtils;

namespace Seq2SeqSharp.Tools
{
    [Serializable]
    public class WeightTensor : IWeightTensor,  IDisposable
    {
        public long[] Sizes { get; set; }

        public int Rows
        {
            get
            {
                return (int)Sizes[0];
            }
            set
            {
                Sizes[0] = value;
            }
        }
        public int Columns
        {
            get
            {
                return (int)Sizes[1];
            }
            set
            {
                Sizes[1] = value;
            }
        }

        public int DeviceId { get; set; }
        IAllocator allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;

        private bool releasedTWeight = false;
        private bool releasedTGradient = false;

        private static object locker = new object();

        public Tensor TWeight
        {
            get
            {
                if (releasedTWeight)
                {
                    return null;
                }

                if (m_TWeight == null)
                {                    
                    m_TWeight = new Tensor(allocator, DType.Float32, Sizes);
                }

                return m_TWeight;
            }
            set
            {
                m_TWeight = value;
                releasedTWeight = false;
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (releasedTGradient)
                {
                    return null;
                }

                if (m_TGradient == null)
                {
                    if (m_TWeight != null)
                    {
                        m_TGradient = new Tensor(allocator, DType.Float32, m_TWeight.Sizes);
                    }
                    else
                    {
                        m_TGradient = new Tensor(allocator, DType.Float32, Sizes);
                    }
                    Ops.Fill(m_TGradient, 0.0f);
                }

                return m_TGradient;
            }

            set
            {
                m_TGradient = value;
                releasedTGradient = false;
            }
        }

        private Tensor m_TCache;
        private bool releasedTCache = false;
     

        public Tensor TCache
        {
            get
            {
                if (releasedTCache)
                {
                    return null;
                }

                if (m_TCache == null)
                {
                    m_TCache = new Tensor(allocator, DType.Float32, Sizes);
                    Ops.Fill(m_TCache, 0.0f);
                }

                return m_TCache;
            }
            set
            {
                m_TCache = value;
                releasedTCache = false;
            }


        }


        public WeightTensor(int rows, int columns, int deviceId, bool normal = false)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Sizes = new long[2];
            Sizes[0] = rows;
            Sizes[1] = columns;

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

            TGradient = new Tensor(allocator, DType.Float32, Sizes);
            Ops.Fill(TGradient, 0.0f);

            TWeight = Tensor.FromArray(allocator, weight).View(Sizes);
        }

        public WeightTensor(int rows, int columns, int deviceId)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Sizes = new long[2];
            Sizes[0] = rows;
            Sizes[1] = columns;
        }

        public WeightTensor(long[] sizes, int deviceId)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Sizes = sizes;
        }

        public WeightTensor(int rows, int columns, Tensor weight, Tensor gradient)
        {
            Sizes = new long[2];
            Sizes[0] = rows;
            Sizes[1] = columns;

            m_TGradient = gradient;
            m_TWeight = weight;
        }


        public WeightTensor(int rows, int columns, float c, int deviceId)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Sizes = new long[2];
            Sizes[0] = rows;
            Sizes[1] = columns;

            var n = rows * columns;

            TGradient = new Tensor(allocator, DType.Float32, Sizes);
            Ops.Fill(TGradient, 0.0f);

            TWeight = new Tensor(allocator, DType.Float32, Sizes);
            Ops.Fill(TWeight, c);
        }


        public void CleanCache()
        {
            Ops.Fill(TCache, 0.0f);
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


        public void SetGradientByWeight(IWeightTensor src)
        {
            WeightTensor m = src as WeightTensor;

            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
            }
            m_TGradient = m.TWeight;

            m.m_TWeight = null;
        }

        public void CopyWeights(IWeightTensor src)
        {
            WeightTensor m = src as WeightTensor;

            Ops.Copy(TWeight, m.TWeight);
        }

        public void AddGradient(IWeightTensor src)
        {
            WeightTensor m = src as WeightTensor;

            lock (locker)
            {
                Tensor t = new Tensor(TGradient.Allocator, DType.Float32, Sizes);
                Ops.Copy(t, m.TGradient);
                Ops.Add(TGradient, TGradient, t);

                t.Dispose();
            }           
        }

        public float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public void AddSoftmaxGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, src.TGradient.Sizes);
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, false);
            }
            else
            {
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight);
            }
        }


        public void CopyOrAddGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, src.TGradient.Sizes);
                Ops.Copy(m_TGradient, src.TGradient);
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src.TGradient);
            }
        }

        public void CopyOrAddGradient(Tensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, src.Sizes);
                Ops.Copy(m_TGradient, src);
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src);
            }
        }

        public void AddMulGradient(Tensor w, Tensor g)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, w.Sizes);
                Ops.Mul(m_TGradient, w, g);
            }
            else
            {
                Ops.AddMul(m_TGradient, m_TGradient, w, g);
            }
        }


        public void AddSigmoidGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, src.TWeight.Sizes);
                Ops.SigmoidD(m_TGradient, src.TWeight, src.TGradient);
            }
            else
            {
                Ops.AddSigmoidD(m_TGradient, m_TGradient, src.TWeight, src.TGradient);
            }
        }


        public void AddTanhGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, src.TWeight.Sizes);

                Ops.TanhD(m_TGradient, src.TWeight, src.TGradient);
            }
            else
            {
                Ops.AddTanhD(m_TGradient, m_TGradient, src.TWeight, src.TGradient);
            }
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

        public List<int> GetTopNMaxWeightIdx(int topN)
        {
            float[] weights = ToWeightArray();
            FixedSizePriorityQueue<ComparableItem<int>> q = new FixedSizePriorityQueue<ComparableItem<int>>(topN, new ComparableItemComparer<int>(true));

            for (int i = 0; i < weights.Length; i++)
            {
                q.Enqueue(new ComparableItem<int>(weights[i], i));
            }

            return q.Select(x => x.Value).ToList();
        }        

        public void SetWeightArray(float[] v)
        {
            TWeight.SetElementsAsFloat(v);
        }

        public void Dispose()
        {
            ReleaseWeight();
            ReleaseGradient();
            ReleaseCache();
        }

        public void ReleaseWeight()
        {
            if (m_TWeight != null)
            {
                m_TWeight.Dispose();
                m_TWeight = null;
                releasedTWeight = true;
            }
        }

        public void ReleaseGradient()
        {
            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
                m_TGradient = null;
                releasedTGradient = true;
            }
        }

        private void ReleaseCache()
        {
            if (m_TCache != null)
            {
                m_TCache.Dispose();
                m_TCache = null;
                releasedTCache = true;
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
