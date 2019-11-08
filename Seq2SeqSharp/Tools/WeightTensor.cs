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

        public string Name { get; set; }
        public bool IsTrainable { get; set; }

        public int DeviceId { get; set; }
        IAllocator m_allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;
        private static object locker = new object();

        public Tensor TWeight
        {
            get
            {
                if (m_TWeight == null)
                {                    
                    m_TWeight = new Tensor(m_allocator, DType.Float32, Sizes);
                }

                return m_TWeight;
            }
            set
            {
                ReleaseWeight();
                m_TWeight = value;
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (m_TGradient == null)
                {
                    if (m_TWeight != null)
                    {
                        m_TGradient = new Tensor(m_allocator, DType.Float32, m_TWeight.Sizes);
                    }
                    else
                    {
                        m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                    }
                    Ops.Fill(m_TGradient, 0.0f);
                }

                return m_TGradient;
            }

            set
            {
                ReleaseGradient();
                m_TGradient = value;
            }
        }
      
        public WeightTensor(long[] sizes, int deviceId, string name = "", bool isTrainable = false, bool normal = false)
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            m_allocator = TensorAllocator.Allocator(DeviceId);
            Sizes = sizes;

            if (normal)
            {
                var n = Rows * Columns;
                float[] weight = new float[n];


                var scale = (float)Math.Sqrt(1.0 / (Rows * Columns));
                if (normal)
                {
                    scale = 0.08f;
                }
                for (int i = 0; i < n; i++)
                {
                    weight[i] = RandomGenerator.NormalRandom(0.0f, scale);
                }

                TWeight = Tensor.FromArray(m_allocator, weight).View(Sizes);
            }
        }

        public WeightTensor(long[] sizes, float c, int deviceId, string name = "", bool isTrainable = false)
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            Sizes = sizes;
            m_allocator = TensorAllocator.Allocator(DeviceId);

            TWeight = new Tensor(m_allocator, DType.Float32, Sizes);
            Ops.Fill(TWeight, c);
        }

        public int GetDeviceId()
        {
            return DeviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new WeightTensor(Sizes, deviceId, Name, IsTrainable);
        }

        public void ZeroGradient()
        {
            Ops.Fill(TGradient, 0.0f);
        }

        public void CleanWeight()
        {
            Ops.Fill(TWeight, 0.0f);
        }

        public float GetWeightAt(int offset)
        {
            return TWeight.GetElementAsFloat(0, offset);
        }

        public void SetWeightAt(float val, int offset)
        {
            TWeight.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            TWeight.SetElementsAsFloat(val, row, 0);
        }

        public void CopyWeightsToGradients(IWeightTensor src)
        {
            WeightTensor m = src as WeightTensor;

            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
            }

            m_TGradient = m.TWeight.CopyRef();
        }

        public void CopyWeightsFrom(IWeightTensor src)
        {
            WeightTensor m = src as WeightTensor;

            Ops.Copy(TWeight, m.TWeight);
        }

        public void AddGradientFrom(IWeightTensor src)
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

        private float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public void AddSoftmaxGradient(WeightTensor src, bool inPlace = false)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, src.TGradient.Sizes);
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, false);
            }
            else
            {
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, !inPlace);
            }
        }

        public void CopyOrAddGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, src.TGradient.Sizes);
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
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, src.Sizes);
                Ops.Copy(m_TGradient, src);
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src);
            }
        }

        public void AddMulGradient(Tensor w, Tensor g, bool inPlace = false)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, w.Sizes);
                Ops.Mul(m_TGradient, w, g);
            }
            else
            {
                if (inPlace)
                {
                    Ops.Mul(m_TGradient, w, g);
                }
                else
                {
                    Ops.AddMul(m_TGradient, m_TGradient, w, g);
                }
            }
        }

        public void AddSigmoidGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, src.TWeight.Sizes);
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
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, src.TWeight.Sizes);

                Ops.TanhD(m_TGradient, src.TWeight, src.TGradient);
            }
            else
            {
                Ops.AddTanhD(m_TGradient, m_TGradient, src.TWeight, src.TGradient);
            }
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

        public WeightTensor CopyWeightsRef(string name)
        {
            WeightTensor result = new WeightTensor(Sizes, DeviceId, name);
            result.m_TWeight = m_TWeight.CopyRef();

            return result;
        }

        public void Dispose()
        {
            ReleaseWeight();
            ReleaseGradient();
        }

        public void ReleaseWeight()
        {
            if (m_TWeight != null)
            {
                m_TWeight.Dispose();
                m_TWeight = null;
            }
        }

        public void ReleaseGradient()
        {
            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
                m_TGradient = null;
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

        public List<IWeightTensor> GetParams()
        {
            return new List<IWeightTensor>() { this };
        }
    }
}
