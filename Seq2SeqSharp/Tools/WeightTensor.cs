// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Tools
{
    public enum RandomInitType
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

        public bool NeedGradient { get; set; }

        public int DeviceId { get; set; }

        public float LearningRateFactor { get; set; } = 1.0f;

        private IAllocator m_allocator;
        public IAllocator Allocator => m_allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;
        private static readonly object locker = new object();
        private readonly IComputeGraph m_computeGraphToBind;

        private string m_GradientSetName = "None";

        private readonly bool m_fanIn = false;
        private readonly bool m_fanOut = false;
        private readonly RandomInitType m_normType = RandomInitType.None;

        private readonly DType m_elementType  =  DType.Float32;

        public long ElementCount
        {
            get
            {
                return m_TWeight.ElementCount();
            }
        }


        public Tensor TWeight
        {
            get
            {
                if (m_TWeight == null)
                {
                    m_TWeight = new Tensor(m_allocator, m_elementType, Sizes);
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
                }
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (m_TGradient == null)
                {
                    m_TGradient = new Tensor(m_allocator, m_elementType, Sizes);
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
                }
            }
        }

        public DType ElementType => m_elementType;

        public WeightTensor(long[] sizes, int deviceId, string name = "", bool isTrainable = false, RandomInitType initType = RandomInitType.None, bool fanIn = false, bool fanOut = false, float learningRateFactor = 1.0f, IComputeGraph graphToBind = null, bool needGradient = true, DType dtype = DType.Float32)
        {
            Name = name;
            DeviceId = deviceId;
            LearningRateFactor = learningRateFactor;
            IsTrainable = isTrainable;
            NeedGradient = needGradient;
            m_allocator = TensorAllocator.Allocator(DeviceId);
            Sizes = sizes;
            m_fanIn = fanIn;
            m_fanOut = fanOut;
            m_normType = initType;
            m_elementType= dtype;

            if (graphToBind != null)
            {
                m_computeGraphToBind = graphToBind;
                m_computeGraphToBind.Bind(this);
            }

            if (isTrainable)
            {
                if (initType == RandomInitType.Uniform)
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
                else if (initType == RandomInitType.Normal)
                {
                    float[] w = TensorSharp.RandomGenerator.BuildRandomUniformWeight(Sizes, -1.0f, 1.0f);
                    SetWeightArray(w);
                }
            }
        }
     

        public WeightTensor(long[] sizes, float c, int deviceId, string name = "", bool isTrainable = false, float learningRateFactor = 1.0f, bool needGradient = true, DType dtype = DType.Float32)
        {
            Name = name;
            DeviceId = deviceId;
            IsTrainable = isTrainable;
            NeedGradient = needGradient;
            LearningRateFactor = learningRateFactor;
            Sizes = sizes;
            m_elementType = dtype;
            m_allocator = TensorAllocator.Allocator(DeviceId);

            var tensor = new Tensor(m_allocator, m_elementType, Sizes);
            Ops.Fill(tensor, c);

            TWeight = tensor;
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
            return new WeightTensor(Sizes, deviceId, Name, IsTrainable, initType: m_normType, fanIn: m_fanIn, fanOut: m_fanOut, needGradient: NeedGradient, dtype: m_elementType);
        }


        public bool IsWeightsCorrupted()
        {
            float[] weights = ToWeightArray();

            for (int i = 0; i < weights.Length; i++)
            {
                if (float.IsNaN(weights[i]) || float.IsInfinity(weights[i]))
                {
                    return true;
                }
            }

            return false;
        }


        public void ZeroGradient()
        {
            Ops.Fill(TGradient, 0.0f);
        }

        public void CleanWeight()
        {
            Ops.Fill(TWeight, 0.0f);
        }

        public void FillGradient(float val)
        {
            Ops.Fill(TGradient, val);
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
                if (m.TGradient.ElementType != m_elementType)
                {
                    throw new DataMisalignedException($"Tensor '{m.Name}' gradient element type is '{m.TGradient.ElementType.ToString()}', but current tensor '{Name}' gradient element type is '{m_elementType}'. They are inconsistent.");
                }

                Tensor t = new Tensor(TGradient.Allocator, m.TGradient.ElementType, Sizes);
                Ops.Copy(t, m.TGradient);
                Ops.Add(TGradient, TGradient, t);

                t.Dispose();
            }
        }

        public float[] ToWeightArray()
        {
            if (ElementType == DType.Float16)
            {
                Tensor tmp = new Tensor(Allocator, DType.Float32, Sizes);
                Ops.Half2Float(tmp, TWeight);
                float[] result = tmp.GetElementsAsFloat((int)tmp.GetStorageSize());
                tmp.Dispose();

                return result;
            }
            else
            {
                return TWeight.GetElementsAsFloat((int)m_TWeight.GetStorageSize());
            }
        }



        static DateTime lastCheckDT = DateTime.Now;
        public void PrintWeights()
        {
            if (DateTime.Now - lastCheckDT >= TimeSpan.FromMinutes(5.0f))
            {
                lastCheckDT = DateTime.Now;
                var weights = ToWeightArray();

                StringBuilder sb = new StringBuilder();
                sb.Append($"Weights for '{Name}': [");

                foreach (var weight in weights)
                {
                    sb.Append($"{weight:F4}, ");
                }

                sb.Append("]");

                Logger.WriteLine(sb.ToString());
            }
        }

        public void AddSoftmaxGradient(WeightTensor src, bool inPlace = false)
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, src.TGradient.ElementType, Sizes);
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
            try
            {
                if (m_TGradient == null)
                {
                    m_allocator = TensorAllocator.Allocator(DeviceId);
                    m_TGradient = new Tensor(m_allocator, src.TGradient.ElementType, Sizes);
                    Ops.Copy(m_TGradient, src.TGradient);

                    m_GradientSetName = "CopyOrAddGradient_WeightTensor";
                }
                else
                {
                    Ops.Add(m_TGradient, m_TGradient, src.TGradient);
                }
            }
            catch(Exception)
            {
                Logger.WriteLine($"Failed to update gradient for tensor '{Name}({TGradient.ToString()}) from '{src.Name}' ({src.TGradient.ToString()})");
                throw;
            }
        }

        public void Clamp(float min, float max)
        {
            Ops.Clamp(TWeight, TWeight, min, max);
        }

        public void CopyOrAddGradient(Tensor src, string callerName = "")
        {
            if (m_TGradient == null)
            {
                m_allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(m_allocator, src.ElementType, Sizes);
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
                m_TGradient = new Tensor(m_allocator, g.ElementType, Sizes);
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
                m_TGradient = new Tensor(m_allocator, src.TGradient.ElementType, Sizes);
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
                m_TGradient = new Tensor(m_allocator, src.TGradient.ElementType, Sizes);

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
            if (ElementType == DType.Float16)
            {
                half[] halves= new half[v.Length];
                for (int i = 0; i < v.Length; i++)
                {
                    halves[i] = new half(v[i]);
                }

                TWeight.SetElementsAsHalf(halves);
            }
            else
            {
                TWeight.SetElementsAsFloat(v);
            }
        }

        public void SetWeightArray(half[] v)
        {
            if (ElementType != DType.Float16)
            {
                throw new InvalidCastException($"Inconsistent element type in weights '{Name}'");
            }

            TWeight.SetElementsAsHalf(v);
        }

        public WeightTensor CopyWeightsRef(string name, bool needGradient, IComputeGraph graphToBind)
        {
            WeightTensor result = new WeightTensor(Sizes, DeviceId, name, needGradient: needGradient, graphToBind: graphToBind, dtype: m_elementType)
            {
                TWeight = TWeight.CopyRef()
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
            Logger.WriteLine($"Loading weights '{Name}' from the model to device '{DeviceId}'...");

            if (m_elementType == DType.Float16)
            {
                var weights = model.GetWeightsHalfType(Name);
                if (weights != null)
                {
                    SetWeightArray(weights);
                }
                else
                {
                    throw new InvalidDataException($"Weights '{Name}' is null.");
                }
            }
            else
            {
                var weights = model.GetWeights(Name);
                if (weights != null)
                {
                    SetWeightArray(weights);
                }
                else
                {
                    throw new InvalidDataException($"Weights '{Name}' is null.");
                }              
            }
        }

        public List<IWeightTensor> GetParams()
        {
            return new List<IWeightTensor>() { this };
        }
    }
}
