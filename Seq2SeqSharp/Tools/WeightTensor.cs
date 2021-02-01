using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;

namespace Seq2SeqSharp.Tools
{
    public enum NormType
    {
        None,
        Uniform,
        Normal
    }

    [Serializable]
    public class WeightTensor : IWeightTensor, IDisposable
    {
        public long[] Sizes { get; set; }

        public int Rows
        {
            get => (int)Sizes[0];
            set => Sizes[0] = value;
        }
        public int Columns
        {
            get => (int)Sizes[1];
            set => Sizes[1] = value;
        }

        public string Name { get; set; }
        public bool IsTrainable { get; set; }

        public int DeviceId { get; set; }

        private IAllocator m_allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;
        private static readonly object locker = new object();

        private bool releasedWeight = false;
        private bool releasedGradient = false;
        private IComputeGraph m_computeGraphToBind;

        private string m_GradientSetName = "None";

        private bool m_fanIn = false;
        private bool m_fanOut = false;
        private NormType m_normType = NormType.None;

        public Tensor TWeight
        {
            get
            {
                if (releasedWeight)
                {
                    throw new Exception($"The weight '{Name}' has been released, you cannot access it.");
                }

                if (m_TWeight == null)
                {
                    m_TWeight = new Tensor(m_allocator, DType.Float32, Sizes);
                }

                return m_TWeight;
            }
            set
            {
                if (m_TWeight != null)
                {
                    throw new Exception($"Please call ReleaseWeight function before assign a new value to weight '{Name}'.");
                }

                m_TWeight = value;

                if (m_TWeight != null)
                {
                    Sizes = m_TWeight.Sizes;
                    if (m_TGradient != null)
                    {
                        for (int i = 0; i < Sizes.Length; i++)
                        {
                            if (Sizes[i] != m_TGradient.Sizes[i])
                            {
                                throw new Exception($"The shape between weights and gradients are different. Name = '{Name}'");
                            }
                        }
                    }
                    releasedWeight = false;
                }
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (releasedGradient)
                {
                    throw new Exception($"The gradient '{Name}' has been released, you cannot access it.");
                }

                if (m_TGradient == null)
                {
                    m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                    Ops.Fill(m_TGradient, 0.0f);

                    m_GradientSetName = "Get";
                }

                return m_TGradient;
            }

            set
            {
                if (m_TGradient != null)
                {                   
                    throw new Exception($"Please call ReleaseGradient function before assign a new value to gradient '{Name}'. This gradient was set by '{m_GradientSetName}'");
                }

                m_TGradient = value;

                if (m_TGradient != null)
                {
                    Sizes = m_TGradient.Sizes;
                    if (m_TWeight != null)
                    {
                        for (int i = 0; i < Sizes.Length; i++)
                        {
                            if (Sizes[i] != m_TWeight.Sizes[i])
                            {
                                throw new Exception($"The shape between weights and gradients are different. Name = '{Name}'");
                            }
                        }
                    }
                    releasedGradient = false;
                }
            }
        }


        public WeightTensor(long[] sizes, int deviceId, string name = "", bool isTrainable = false, NormType normType = NormType.None, bool fanIn = false, bool fanOut = false, IComputeGraph graphToBind = null)
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            m_allocator = TensorAllocator.Allocator(DeviceId);
            Sizes = sizes;
            m_fanIn = fanIn;
            m_fanOut = fanOut;
            m_normType = normType;

            if (graphToBind != null)
            {
                m_computeGraphToBind = graphToBind;
                m_computeGraphToBind.Bind(this);
            }

            if (normType == NormType.Uniform)
            {
                var scale = (float)Math.Sqrt(6.0 / (double)(Rows + Columns));

                if (fanIn && !fanOut)
                {
                    scale = (float)Math.Sqrt(3.0 / (double)Rows);
                }
                else if (!fanIn && fanOut)
                {
                    scale = (float)Math.Sqrt(3.0 / (double)Columns);
                }

                float[] w = TensorSharp.RandomGenerator.BuildRandomUniformWeight(Sizes, -scale, scale);
                SetWeightArray(w);               
            }
            else if (normType == NormType.Normal)
            {
                float[] w = TensorSharp.RandomGenerator.BuildRandomUniformWeight(Sizes, -1.0f, 1.0f);
                SetWeightArray(w);
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


        public void UnbindFromComputeGraph()
        {
            if (m_computeGraphToBind != null)
            {
                m_computeGraphToBind.Unbind(this);
            }
        }

        public int GetDeviceId()
        {
            return DeviceId;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new WeightTensor(Sizes, deviceId, Name, IsTrainable, normType: m_normType, fanIn: m_fanIn, fanOut: m_fanOut);
        }

        public void ZeroGradient()
        {
            Ops.Fill(TGradient, 0.0f);
        }

        public void CleanWeight()
        {
            Ops.Fill(TWeight, 0.0f);
        }

        public float GetWeightAt(long[] indices)
        {
            return TWeight.GetElementAsFloat(indices);
        }

        public void SetWeightAt(float val, long[] indices)
        {
            TWeight.SetElementAsFloat(val, indices);
        }


        public void SetGradientAt(float val, long[] indices)
        {
            TGradient.SetElementAsFloat(val, indices);
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

            m_GradientSetName = "CopyWeightsToGradients";
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

        public float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public float[] ToGradientArray()
        {
            return TGradient.GetElementsAsFloat(Rows * Columns);
        }


        public void AddSoftmaxGradient(WeightTensor src, bool inPlace = false)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, false);

                releasedGradient = false;

                m_GradientSetName = "AddSoftmaxGradient";
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
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.Copy(m_TGradient, src.TGradient);

                releasedGradient = false;

                m_GradientSetName = "CopyOrAddGradient_WeightTensor";
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src.TGradient);
            }
        }

        public void CopyOrAddGradient(Tensor src, string callerName = "")
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.Copy(m_TGradient, src);

                releasedGradient = false;

                m_GradientSetName = $"CopyOrAddGradient_Tensor_CalledBy_{callerName}";
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
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.Mul(m_TGradient, w, g);

                releasedGradient = false;

                m_GradientSetName = "AddMulGrdient";
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
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.SigmoidD(m_TGradient, src.TWeight, src.TGradient);

                releasedGradient = false;

                m_GradientSetName = "AddSigmoidGradient";
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
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);

                Ops.TanhD(m_TGradient, src.TWeight, src.TGradient);

                releasedGradient = false;

                m_GradientSetName = "AddTanhGradient";
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

        public void SetGradientArray(float[] v)
        {
            TGradient.SetElementsAsFloat(v);
        }

        public WeightTensor CopyWeightsRef(string name)
        {
            WeightTensor result = new WeightTensor(Sizes, DeviceId, name)
            {
                m_TWeight = m_TWeight.CopyRef()
            };

            return result;
        }

        public void Dispose()
        {
            ReleaseWeight();
            ReleaseGradient();
        }

        public bool IsWeightNull()
        {
            return m_TWeight == null;
        }

        public bool IsGradientNull()
        {
            return m_TGradient == null;
        }

        public void ReleaseWeight()
        {
            if (m_TWeight != null)
            {
                m_TWeight.Dispose();
                m_TWeight = null;
                releasedWeight = true;
            }
        }

        public void ReleaseGradient()
        {
            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
                m_TGradient = null;
                releasedGradient = true;
            }
        }

        public void Save(Stream stream)
        {
            float[] floatArray1 = ToWeightArray();

            // create a byte array and copy the floats into it...
            byte[] byteArray = new byte[floatArray1.Length * 4];
            Buffer.BlockCopy(floatArray1, 0, byteArray, 0, byteArray.Length);

            stream.Write(byteArray, 0, byteArray.Length);
        }

        public void Load(Stream stream)
        {
            int size = Rows * Columns;
            byte[] byteArray = new byte[size * 4];
            stream.Read(byteArray, 0, byteArray.Length);

            float[] floatArray2 = new float[byteArray.Length / 4];
            Buffer.BlockCopy(byteArray, 0, floatArray2, 0, byteArray.Length);

            SetWeightArray(floatArray2);
        }

        public List<IWeightTensor> GetParams()
        {
            if (IsTrainable)
            {
                return new List<IWeightTensor>() { this };
            }
            else
            {
                return new List<IWeightTensor>();
            }
        }
    }
}
