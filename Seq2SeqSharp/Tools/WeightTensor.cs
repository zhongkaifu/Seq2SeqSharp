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

        public float LearningRateFactor { get; set; } = 1.0f;

        private IAllocator m_allocator;
        public IAllocator Allocator => m_allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;
        private static readonly object locker = new object();

        private bool releasedWeight = false;
      //  private bool releasedGradient = false;
        private readonly IComputeGraph m_computeGraphToBind;

        private string m_GradientSetName = "None";

        private readonly bool m_fanIn = false;
        private readonly bool m_fanOut = false;
        private readonly NormType m_normType = NormType.None;

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
                //if (releasedGradient)
                //{
                //    throw new Exception($"The gradient '{Name}' has been released, you cannot access it.");
                //}

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
                    //releasedGradient = false;
                }
            }
        }


        public WeightTensor(long[] sizes, int deviceId, string name = "", bool isTrainable = false, NormType normType = NormType.None, bool fanIn = false, bool fanOut = false, float learningRateFactor = 1.0f, IComputeGraph graphToBind = null)
        {
            Name = name;
            DeviceId = deviceId;
            LearningRateFactor = learningRateFactor;
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

        public WeightTensor(long[] sizes, float c, int deviceId, string name = "", bool isTrainable = false, float learningRateFactor = 1.0f)
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            LearningRateFactor = learningRateFactor;
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

        public float GetGradientAt(long[] indices)
        {
            return TGradient.GetElementAsFloat(indices);
        }

        public void SetWeightAt(float val, long[] indices)
        {
            TWeight.SetElementAsFloat(val, indices);
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

        public void AddSoftmaxGradient(WeightTensor src, bool inPlace = false)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, DType.Float32, Sizes);
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, false);

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
            }
        }

        public void Save(IModel model)
        {
            model.AddWeights(Name, ToWeightArray());
        }

        public void Load(IModel model)
        {
            Logger.WriteLine($"Loading weights '{Name}' from the model...");

            var weights = model.GetWeights(Name);
            if (weights != null)
            {
                SetWeightArray(weights);
            }
        }

        public List<IWeightTensor> GetParams()
        {
            return new List<IWeightTensor>() { this };
        }
    }
}
