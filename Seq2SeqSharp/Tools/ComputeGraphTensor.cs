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
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

/// <summary>
/// Tensor based computing graph written by Zhongkai Fu.
/// The graph includes the following key features.
/// #1. Include several key operations for neural network.
/// #2. Support both CPU and GPU (CUDA)
/// #3. Support automatic differentiation and back propagation
/// #4. Support networks (operations) visualization
/// #5. and so on...
/// </summary>
namespace Seq2SeqSharp.Tools
{
    public class ConcurrentList<T>
    {
        private const int MaxSize = 1024000;
        private T[] array;
        private int count = 0;
        public int Count => count;

        public T this[int key]
        {
            get => array[key];
            set => array[key] = value;
        }

        public ConcurrentList()
        {
            array = new T[MaxSize];
        }

        public void Add(T item)
        {
            int n = System.Threading.Interlocked.Increment(ref count);
            array[n - 1] = item;
        }

        public void RemoveLastItem()
        {
            System.Threading.Interlocked.Decrement(ref count);
        }

        private readonly object locker = new object();
        public void Clear()
        {
            lock (locker)
            {
                count = 0;
                array = new T[MaxSize];
            }
        }
    }

    public class ComputeGraphTensor : IComputeGraph
    {
        private readonly WeightTensorFactory m_weightTensorFactory;
        private readonly ConcurrentList<Action> m_backprop;
        private readonly bool m_needsBackprop;
      //  private readonly bool m_visNeuralNetwork;
        private readonly int m_deviceId;
        private readonly bool m_isSubGraph;

        // Visualization for neural network
       // private Microsoft.Msagl.Drawing.Graph m_opsViz;
       // private HashSet<string> m_setEdges;
        //private Microsoft.Msagl.Drawing.Subgraph m_subGraph = null;
       // private Dictionary<string, Microsoft.Msagl.Drawing.Subgraph> m_name2SubGraph = null;

        private readonly List<IWeightTensor> m_tensorsBindToCurrentGraph;

        public ComputeGraphTensor(IWeightFactory weightFactory, int deviceId, bool needBack = true, ConcurrentList<Action> backprop = null, bool isSubGraph = false)
        {
            m_backprop = backprop ?? new ConcurrentList<Action>();
            m_weightTensorFactory = weightFactory as WeightTensorFactory;
            m_needsBackprop = needBack;
            m_deviceId = deviceId;
            //m_visNeuralNetwork = visNetwork;
            m_isSubGraph = isSubGraph;

            //m_name2SubGraph = new Dictionary<string, Subgraph>();
            //if (m_visNeuralNetwork)
            //{
            //    // Initialize parameters for neural network visualization
            //    m_opsViz = new Microsoft.Msagl.Drawing.Graph();
            //    m_setEdges = new HashSet<string>();
            //}

            m_tensorsBindToCurrentGraph = new List<IWeightTensor>();
        }

        public IWeightFactory GetWeightFactory()
        {
            return m_weightTensorFactory;
        }

        public IComputeGraph CreateSubGraph(string name)
        {
            ComputeGraphTensor subGraph = new ComputeGraphTensor(m_weightTensorFactory, m_deviceId, m_needsBackprop, m_backprop, isSubGraph: true);
            //if (m_visNeuralNetwork)
            //{
            //    // Create parameters for neural network visualization
            //    subGraph.m_opsViz = m_opsViz;
            //    subGraph.m_setEdges = m_setEdges;
            //    subGraph.m_name2SubGraph = m_name2SubGraph;
            //    if (m_name2SubGraph.ContainsKey(name) == false)
            //    {
            //        int index = name.LastIndexOf(".");
            //        subGraph.m_subGraph = new Subgraph(name)
            //        {
            //            LabelText = name.Substring(index + 1)
            //        };

            //        m_name2SubGraph.Add(name, subGraph.m_subGraph);

            //        if (m_subGraph == null)
            //        {
            //            m_opsViz.RootSubgraph.AddSubgraph(subGraph.m_subGraph);
            //        }
            //        else
            //        {
            //            m_subGraph.AddSubgraph(subGraph.m_subGraph);
            //        }
            //    }
            //    else
            //    {
            //        subGraph.m_subGraph = m_name2SubGraph[name];
            //    }
            //}

            return subGraph;
        }

        public void Backward()
        {
            for (int i = m_backprop.Count - 1; i >= 0; i--)
            {
                m_backprop[i](); // tick!
            }

            m_backprop.Clear();
        }

        public IWeightTensor Sigmoid(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Sigmoid", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            Ops.Sigmoid(res.TWeight, m.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        m.AddSigmoidGradient(res);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }



        public IWeightTensor Rsqrt(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Rsqrt", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            Ops.Rsqrt(res.TWeight, m.TWeight);
          
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        using (var tmp = Ops.Pow(null, res.TWeight, 3.0f))
                        {
                            using var tmp2 = Ops.Mul(null, tmp, res.TGradient);
                            using var tmp3 = Ops.Mul(null, tmp2, -0.5f);
                            m.CopyOrAddGradient(tmp3);

                        }
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }



        public IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor m2 = w2 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.AddTanh", graphToBind: this, needGradient: (m1.NeedGradient || m2.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.AddTanh(res.TWeight, m1.TWeight, m2.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m1.NeedGradient)
                    {
                        m1.AddTanhGradient(res);
                    }

                    if (m2.NeedGradient)
                    {
                        m2.AddTanhGradient(res);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor m2 = w2 as WeightTensor;
            WeightTensor m3 = w3 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name, w3.Name)}.AddTanh", graphToBind: this, needGradient: (m1.NeedGradient || m2.NeedGradient || m3.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { w1, w2, w3 }, res);

            Ops.AddTanh3(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m1.NeedGradient)
                    {
                        m1.AddTanhGradient(res);
                    }

                    if (m2.NeedGradient)
                    {
                        m2.AddTanhGradient(res);
                    }

                    if (m3.NeedGradient)
                    {
                        m3.AddTanhGradient(res);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor Mul(IWeightTensor w, float v, bool inPlace = false)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = null;

            if (inPlace)
            {
                res = m.CopyWeightsRef($"{GetHashString(m.Name)}.MulV", m.NeedGradient);
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.MulV", graphToBind: this, needGradient: m.NeedGradient);
            }

            VisualizeNodes(w, res);

            Ops.Mul(res.TWeight, m.TWeight, v);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();

                        if (inPlace && res.TGradient.IsOwnerExclusive() && m.IsGradientNull())
                        {
                            m.TGradient = res.TGradient.CopyRef();
                            Ops.Mul(m.TGradient, res.TGradient, v);
                        }
                        else
                        {
                            Ops.AddMulV(m.TGradient, m.TGradient, res.TGradient, v);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Div(IWeightTensor w, float v, bool inPlace = false)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = null;

            if (inPlace)
            {
                res = m.CopyWeightsRef($"{GetHashString(m.Name)}.DivV", m.NeedGradient);
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.DivV", graphToBind: this, needGradient: m.NeedGradient);
            }

            VisualizeNodes(w, res);

            Ops.Div(res.TWeight, m.TWeight, v);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();

                        if (inPlace && res.TGradient.IsOwnerExclusive() && m.IsGradientNull())
                        {
                            m.TGradient = res.TGradient.CopyRef();
                            Ops.Div(m.TGradient, res.TGradient, v);
                        }
                        else
                        {
                            Ops.AddMulV(m.TGradient, m.TGradient, res.TGradient, 1.0f / v);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public void Bind(IWeightTensor w)
        {
            m_tensorsBindToCurrentGraph.Add(w);
        }

        public void Unbind(IWeightTensor w)
        {
            m_tensorsBindToCurrentGraph.Remove(w);

        }

        /// <summary>
        /// Result = w1 * w2 + w3 * w4
        /// </summary>
        /// <param name="w1"></param>
        /// <param name="w2"></param>
        /// <param name="w3"></param>
        /// <param name="w4"></param>
        /// <returns></returns>
        public IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor m2 = w2 as WeightTensor;
            WeightTensor m3 = w3 as WeightTensor;
            WeightTensor m4 = w4 as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name, w3.Name, w4.Name)}.EltMulMulAdd", graphToBind: this, needGradient: (m1.NeedGradient || m2.NeedGradient || m3.NeedGradient || m4.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { w1, w2, w3, w4 }, res);

            Ops.MulMulAdd(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (m1.NeedGradient)
                    {
                        m1.AddMulGradient(m2.TWeight, res.TGradient);
                    }

                    if (m2.NeedGradient)
                    {
                        m2.AddMulGradient(m1.TWeight, res.TGradient);
                    }

                    if (m3.NeedGradient)
                    {
                        m3.AddMulGradient(m4.TWeight, res.TGradient);
                    }

                    if (m4.NeedGradient)
                    {
                        m4.AddMulGradient(m3.TWeight, res.TGradient);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);

                // These tensors' weights will be used during back-propogation, so we unbind them from the computing graph
                m1.UnbindFromComputeGraph();
                m2.UnbindFromComputeGraph();
                m3.UnbindFromComputeGraph();
                m4.UnbindFromComputeGraph();
            }


            return res;
        }

        public IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor m2 = w2 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.EltMul", graphToBind: this, needGradient: (m1.NeedGradient || m2.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.Mul(res.TWeight, m1.TWeight, m2.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (m1.NeedGradient)
                    {
                        m1.AddMulGradient(m2.TWeight, res.TGradient);
                    }

                    if (m2.NeedGradient)
                    {
                        m2.AddMulGradient(m1.TWeight, res.TGradient);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);

                m1.UnbindFromComputeGraph();
                m2.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Add(IWeightTensor w1, IWeightTensor w2, bool inPlace = false)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor m2 = w2 as WeightTensor;
            WeightTensor res = null;

            if (inPlace)
            {
                res = m1.CopyWeightsRef($"{GetHashString(w1.Name)}.Add", needGradient: (m1.NeedGradient || m2.NeedGradient));
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.Add", graphToBind: this, needGradient: (m1.NeedGradient || m2.NeedGradient));
            }

            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);


            Ops.Add(res.TWeight, m1.TWeight, m2.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (m1.NeedGradient)
                    {
                        if (res.TGradient.IsOwnerExclusive() && m1.IsGradientNull())
                        {
                            m1.TGradient = res.TGradient.CopyRef();
                        }
                        else
                        {
                            m1.CopyOrAddGradient(res);
                        }
                    }

                    if (m2.NeedGradient)
                    {
                        if (res.TGradient.IsOwnerExclusive() && m2.IsGradientNull())
                        {
                            m2.TGradient = res.TGradient.CopyRef();
                        }
                        else
                        {
                            m2.CopyOrAddGradient(res);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Sum(IWeightTensor w, int dim)
        {
            WeightTensor m = w as WeightTensor;
            var newSizes = (long[])m.Sizes.Clone();
            newSizes[dim] = 1;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(newSizes, m_deviceId, name: $"{m.Name}.Sum", graphToBind: this, needGradient: m.NeedGradient);
            Ops.Sum(res.TWeight, m.TWeight, dim);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        using var tmp = res.TGradient.Expand(m.Sizes);
                        m.CopyOrAddGradient(tmp);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor Log(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(m.Name)}.Log", graphToBind: this, needGradient: m.NeedGradient);

            Ops.Log(res.TWeight, m.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        Ops.AddDiv(m.TGradient, m.TGradient, res.TGradient, m.TWeight);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);

                m.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Add(IWeightTensor w1, float v)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name)}.AddTV", graphToBind: this, needGradient: m1.NeedGradient);

            VisualizeNodes(new IWeightTensor[] { w1}, res);


            Ops.Add(res.TWeight, m1.TWeight, v);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m1.NeedGradient)
                    {
                        res.ReleaseWeight();
                        if (res.TGradient.IsOwnerExclusive() && m1.IsGradientNull())
                        {
                            m1.TGradient = res.TGradient.CopyRef();
                        }
                        else
                        {
                            m1.CopyOrAddGradient(res);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }



        public IWeightTensor Sub(float v, IWeightTensor w1)
        {
            WeightTensor m1 = w1 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name)}.SubVT", graphToBind: this, needGradient: m1.NeedGradient);

            VisualizeNodes(new IWeightTensor[] { w1 }, res);

            Ops.Sub(res.TWeight, v, m1.TWeight);         

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m1.NeedGradient)
                    {
                        res.ReleaseWeight();
                        Ops.Sub(m1.TGradient, m1.TGradient, res.TGradient);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }



        public IWeightTensor Tanh(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Tanh", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            Ops.Tanh(res.TWeight, m.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        m.AddTanhGradient(res);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Relu(IWeightTensor w, bool inPlace = false)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = null;
            if (inPlace)
            {
                res = m.CopyWeightsRef($"{GetHashString(w.Name)}.Relu", needGradient: m.NeedGradient);
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Relu", graphToBind: this, needGradient: m.NeedGradient);
            }
            VisualizeNodes(w, res);


            Ops.Relu(res.TWeight, m.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();

                        if (inPlace && m.IsGradientNull() && res.TGradient.IsOwnerExclusive())
                        {
                            m.TGradient = res.TGradient.CopyRef();
                            Ops.ReluD(m.TGradient, m.TWeight, m.TGradient);
                        }
                        else
                        {
                            Ops.AddReluD(m.TGradient, m.TGradient, m.TWeight, res.TGradient);
                        }
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);

                m.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, float alpha = 1.0f)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { t1.TWeight.Sizes[0], t1.TWeight.Sizes[1], t2.TWeight.Sizes[2] }, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name)}.MulBatch", graphToBind: this, needGradient: (t1.NeedGradient || t2.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            Ops.AddmmBatch(res.TWeight, 0.0f, res.TWeight, alpha, t1.TWeight, t2.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (t1.NeedGradient)
                    {
                        using Tensor tW2 = t2.TWeight.Transpose(1, 2);
                        Ops.AddmmBatch(t1.TGradient, 1.0f, t1.TGradient, alpha, res.TGradient, tW2);
                    }

                    if (t2.NeedGradient)
                    {
                        using Tensor tW1 = t1.TWeight.Transpose(1, 2);
                        Ops.AddmmBatch(t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, res.TGradient);
                    }

                    res.Dispose();

                }
                m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Mul(IWeightTensor m1, IWeightTensor m2, float alpha = 1.0f)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            int n = t1.Rows;
            int d = t2.Columns;
            WeightTensor res;

            res = m_weightTensorFactory.CreateWeightTensor(n, d, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name)}.Mul", graphToBind: this, needGradient: (t1.NeedGradient || t2.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            Ops.Addmm(res.TWeight, 0.0f, res.TWeight, alpha, t1.TWeight, t2.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (t1.NeedGradient)
                    {
                        using Tensor tW2 = t2.TWeight.Transpose();
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, alpha, res.TGradient, tW2);
                    }

                    if (t2.NeedGradient)
                    {
                        using Tensor tW1 = t1.TWeight.Transpose();
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, res.TGradient);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;
        }

        public IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias, float alpha = 1.0f)
        {
            if (m1 == null)
            {
                ArgumentNullException argumentNullException = new ArgumentNullException($"m1 tensor is null");
                throw argumentNullException;
            }

            if (m2 == null)
            {
                ArgumentNullException argumentNullException = new ArgumentNullException($"m2 tensor is null");
                throw argumentNullException;
            }

            if (mbias == null)
            {
                ArgumentNullException argumentNullException = new ArgumentNullException($"mbias tensor is null");
                throw argumentNullException;
            }

            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = mbias as WeightTensor;

            int n = t1.Rows;
            int d = t2.Columns;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(n, d, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name, mbias.Name)}.Affine", graphToBind: this, needGradient: (t1.NeedGradient || t2.NeedGradient || t3.NeedGradient));
            VisualizeNodes(new IWeightTensor[] { m1, m2, mbias }, res);

            using (Tensor t3WExp = t3.TWeight.Expand(n, d))
            {
                Ops.Addmm(res.TWeight, 1.0f, t3WExp, alpha, t1.TWeight, t2.TWeight);
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();

                    if (t3.NeedGradient)
                    {
                        using Tensor t3G = t3.TGradient.Expand(n, d);
                        Ops.Add(t3G, t3G, res.TGradient);
                    }

                    if (t2.NeedGradient)
                    {
                        using Tensor tW2 = t2.TWeight.Transpose();
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, alpha, res.TGradient, tW2);
                    }

                    if (t1.NeedGradient)
                    {
                        using Tensor tW1 = t1.TWeight.Transpose();
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, alpha, tW1, res.TGradient);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);

                t1.UnbindFromComputeGraph();
                t2.UnbindFromComputeGraph();
            }

            return res;

        }

        public IWeightTensor Transpose(IWeightTensor w, int dim1, int dim2)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Transpose", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Transpose(dim1, dim2);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        bool isOwnerExclusive = res.TGradient.IsOwnerExclusive();
                        using Tensor gT = res.TGradient.Transpose(dim1, dim2);
                        if (isOwnerExclusive && m.IsGradientNull())
                        {
                            m.TGradient = gT.CopyRef();
                        }
                        else
                        {
                            m.CopyOrAddGradient(gT, res.Name);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Transpose(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Columns, m.Rows, m_deviceId, name: $"{GetHashString(w.Name)}.Transpose", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Transpose();
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        bool isOwnerExclusive = res.TGradient.IsOwnerExclusive();

                        using Tensor gT = res.TGradient.Transpose();
                        if (isOwnerExclusive && m.IsGradientNull())
                        {
                            m.TGradient = gT.CopyRef();
                        }
                        else
                        {
                            m.CopyOrAddGradient(gT, res.Name);
                        }
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Argmax(IWeightTensor w, int dim)
        {
            WeightTensor m = w as WeightTensor;
            Tensor argMaxT = Ops.Argmax(null, m.TWeight, dim);

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(argMaxT.Sizes, m_deviceId, name: $"{GetHashString(m.Name)}.Argmax", graphToBind: this, needGradient: m.NeedGradient);
            res.TWeight = argMaxT;

            if (m_needsBackprop)
            {
                throw new NotSupportedException($"Argmax operation doesn't support back propagation.");
            }

            return res;
        }

        private static double PowerA(double a, double b)
        {
            int tmp = (int)(BitConverter.DoubleToInt64Bits(a) >> 32);
            int tmp2 = (int)(b * (tmp - 1072632447) + 1072632447);
            return BitConverter.Int64BitsToDouble(((long)tmp2) << 32);
        }

        /// <summary>
        /// Top-P sampling for each row in given tensor
        /// </summary>
        /// <param name="w"></param>
        /// <param name="seqs"></param>
        /// <param name="topP"></param>
        /// <returns>The sampled index</returns>
        public IWeightTensor TopPSampleIndice(IWeightTensor w, List<List<int>> seqs, float topP = 0.95f, float repeatPenalty = 5.0f)
        {
            WeightTensor m = w as WeightTensor;
            float[] weights = m.ToWeightArray();
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { m.Rows, 1}, m_deviceId, name: $"{GetHashString(m.Name)}.Sample", graphToBind: this, needGradient: m.NeedGradient);

            object locker = new object();
            Random rnd = new Random(DateTime.Now.Millisecond);
            float[] indices = new float[m.Rows];
            float thresholdValue = 1.0f / (float)(m.Columns * 10000.0);

            for (int i = 0; i < m.Rows; i++)
            {
                int offset = i * m.Columns;
                List<int> seq = seqs[i];

                Dictionary<int, int> tokenId2OffsetInSeq = new Dictionary<int, int>(); // <tokenId, offsetInSeq>. The last offset of the token in the given sequence
                Dictionary<int, int> tokenId2Cnt = new Dictionary<int, int>(); // <tokenId, count> The number of token in the given sequences
                for (int j = 0; j < seq.Count; j++)
                {
                    if (tokenId2OffsetInSeq.ContainsKey(seq[j]) == false)
                    {
                        tokenId2OffsetInSeq.Add(seq[j], j);
                        tokenId2Cnt.Add(seq[j], 0);
                    }
                    else
                    {
                        tokenId2OffsetInSeq[seq[j]] = j;
                    }

                    tokenId2Cnt[seq[j]]++;
                }


                if (topP == 0.0f)
                {
                    float maxWeight = float.MinValue;
                    int maxWeightIndice = -1;

                    for (int j = 0; j < m.Columns; j++)
                    {
                        float weight = weights[offset + j];
                        if (Math.Abs(weight) < thresholdValue)
                        {
                            continue;
                        }

                        //Decay weights if tokens has already been generated before
                        if (tokenId2OffsetInSeq.ContainsKey(j))
                        {
                            int offsetInSeq = tokenId2OffsetInSeq[j];
                            weight = (float)(weight * PowerA(repeatPenalty, -1.0 - (double)tokenId2Cnt[j] / (double)(seq.Count - offsetInSeq)));
                        }

                        if (Math.Abs(weight) < thresholdValue)
                        {
                            continue;
                        }

                        if (weight > maxWeight)
                        {
                            maxWeight = weight;
                            maxWeightIndice = j;
                        }
                    }

                    indices[i] = maxWeightIndice;
                }
                else
                {
                    SortedDictionary<float, int> weight2tokenId = new SortedDictionary<float, int>();
                    float adjustedSum = 0.0f;
                    for (int j = 0; j < m.Columns; j++)
                    {
                        float weight = weights[offset + j];
                        if (Math.Abs(weight) < thresholdValue || weight2tokenId.ContainsKey(weight))
                        {
                            continue;
                        }

                        //Decay weights if tokens has already been generated before
                        if (tokenId2OffsetInSeq.ContainsKey(j))
                        {
                            int offsetInSeq = tokenId2OffsetInSeq[j];
                            weight = (float)(weight * PowerA(repeatPenalty, -1.0 - (double)tokenId2Cnt[j] / (double)(seq.Count - offsetInSeq)));
                        }

                        if (Math.Abs(weight) < thresholdValue || weight2tokenId.ContainsKey(weight))
                        {
                            continue;
                        }

                        adjustedSum += weight;
                        weight2tokenId.Add(weight, j);

                    }


                    float acc = 0.0f;
                    float seed = 0.0f;
                    lock (locker)
                    {
                        seed = (float)rnd.NextDouble() * topP * adjustedSum;
                    }

                    foreach (var pair in weight2tokenId.Reverse())
                    {
                        acc += pair.Key;

                        if (acc >= seed)
                        {
                            indices[i] = pair.Value;
                            break;
                        }
                    }
                }
            }

            res.SetWeightArray(indices);


            if (m_needsBackprop)
            {
                throw new NotSupportedException($"TopPSampleIndice operation doesn't support back propagation.");
            }


            return res;

        }


        public IWeightTensor Max(IWeightTensor w, int dim)
        {
            WeightTensor m = w as WeightTensor;
            Tensor argMaxT = Ops.Max(null, m.TWeight, dim);

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(argMaxT.Sizes, m_deviceId, name: $"{GetHashString(m.Name)}.Max", graphToBind: this, needGradient: m.NeedGradient);
            res.TWeight = argMaxT;

            if (m_needsBackprop)
            {
                throw new NotSupportedException($"Argmax operation doesn't support back propagation.");
            }

            return res;
        }


        public IWeightTensor Softmax(IWeightTensor w, bool runGradients = true, bool inPlace = false)
        {
            WeightTensor t = w as WeightTensor;
            WeightTensor res = null;

            if (inPlace)
            {
                res = t.CopyWeightsRef($"{GetHashString(w.Name)}.Softmax", needGradient: runGradients && t.NeedGradient);
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(t.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Softmax", graphToBind: this, needGradient: runGradients && t.NeedGradient);
            }

            VisualizeNodes(w, res);

            Ops.Softmax(res.TWeight, t.TWeight);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (runGradients && t.NeedGradient)
                    {
                        if (inPlace && t.IsGradientNull() && res.TGradient.IsOwnerExclusive())
                        {
                            t.TGradient = res.TGradient.CopyRef();
                        }
                        t.AddSoftmaxGradient(res, inPlace);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);

                res.UnbindFromComputeGraph();

            }

            return res;
        }

        public IWeightTensor Peek(IWeightTensor w, int dim, int ix, int num = 1)
        {
            WeightTensor m = w as WeightTensor;

            long[] sizes = (long[])m.Sizes.Clone();
            sizes[dim] = num;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Peek", graphToBind: this, needGradient: m.NeedGradient);
            res.TWeight = m.TWeight.Narrow(dim, ix, num);
            res.TGradient = res.NeedGradient ? m.TGradient.Narrow(dim, ix, num) : null;

            VisualizeNodes(w, res);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        private string GetHashString(params string[] inputStrings)
        {
            //if (m_visNeuralNetwork)
            //{
            //    string inputString = string.Join("_", inputStrings);
            //    StringBuilder sb = new StringBuilder();
            //    foreach (byte b in GetHash(inputString))
            //    {
            //        sb.Append(b.ToString("X2"));
            //    }

            //    return sb.ToString();
            //}
            return string.Empty;
        }

        private void VisualizeNodes(IWeightTensor sourceNode, IWeightTensor targetNode)
        {
            VisualizeNodes(new IWeightTensor[] { sourceNode }, targetNode);
        }

        private void VisualizeNodes(IEnumerable<IWeightTensor> sourceNodes, IWeightTensor targetNode)
        {
            //if (!m_visNeuralNetwork || m_deviceId != 0)
            //{
            //    return;
            //}

            //// Create node for target tensor
            //int index = targetNode.Name.LastIndexOf('.');
            //Microsoft.Msagl.Drawing.Node tgtNode = m_opsViz.AddNode(targetNode.Name);
            //tgtNode.LabelText = targetNode.Name.Substring(index + 1);

            //if (targetNode.IsTrainable)
            //{
            //    tgtNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;
            //}

            //if (m_subGraph != null)
            //{
            //    // Current compute graph is a sub-graph
            //    m_subGraph.AddNode(tgtNode);
            //}

            //// Create edges for each source node and target node
            //foreach (IWeightTensor sourceNode in sourceNodes)
            //{
            //    if (!sourceNode.Name.IsNullOrEmpty() && !targetNode.Name.IsNullOrEmpty())
            //    {
            //        string key = $"{sourceNode.Name}->{targetNode.Name}";
            //        if (m_setEdges.Contains(key))
            //        {
            //            continue;
            //        }

            //        int srcIndex = sourceNode.Name.LastIndexOf('.');
            //        Microsoft.Msagl.Drawing.Node srcNode = m_opsViz.AddNode(sourceNode.Name);
            //        srcNode.LabelText = sourceNode.Name.Substring(srcIndex + 1);
            //        if (sourceNode.IsTrainable)
            //        {
            //            srcNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;

            //            if (m_subGraph != null)
            //            {
            //                m_subGraph.AddNode(srcNode);
            //            }
            //        }

            //        Edge edge = m_opsViz.AddEdge(sourceNode.Name, targetNode.Name);

            //        m_setEdges.Add(key);
            //    }
            //}
        }

        public void VisualizeNeuralNetToFile(string neuralNetPicFilePath)
        {
            //FastIncrementalLayoutSettings fastSettings = new FastIncrementalLayoutSettings
            //{
            //    AvoidOverlaps = true,
            //    NodeSeparation = 30,
            //    RouteEdges = true
            //};

            //SugiyamaLayoutSettings settings = new SugiyamaLayoutSettings
            //{
            //    FallbackLayoutSettings = fastSettings
            //};

            //m_opsViz.LayoutAlgorithmSettings = settings;

            //Microsoft.Msagl.GraphViewerGdi.GraphRenderer renderer = new Microsoft.Msagl.GraphViewerGdi.GraphRenderer(m_opsViz);
            //renderer.CalculateLayout();

            //System.Drawing.Bitmap bitmap = new System.Drawing.Bitmap((int)m_opsViz.Width, (int)m_opsViz.Height, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            //renderer.Render(bitmap);

            //bitmap.Save(neuralNetPicFilePath);

            //bitmap.Dispose();
        }


        public IWeightTensor IndexSelect(IWeightTensor s, float[] idxs, bool clearWeights = false)
        {
            WeightTensor src = s as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { idxs.Length, s.Sizes[^1] }, m_deviceId, name: $"{GetHashString(src.Name)}.IndexSelect", graphToBind: this, needGradient: src.NeedGradient, clearWeights: clearWeights);

            Tensor indice = new Tensor(src.Allocator, DType.Float32, idxs.Length);
            indice.CopyFrom(idxs);
            Ops.IndexSelect(res.TWeight, src.TWeight, indice);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (src.NeedGradient)
                    {
                        res.ReleaseWeight();
                        Ops.IndexSelectGrad(src.TGradient, res.TGradient, indice);
                    }

                    res.Dispose();
                    indice.Dispose();
                }
                m_backprop.Add(backward);
            }
            else
            {
                indice.Dispose();
            }

            return res;
        }

        public IWeightTensor Concate(int dim, params IWeightTensor[] wl)
        {
            return Concate(wl.ToList(), dim);
        }

        public IWeightTensor Concate(List<IWeightTensor> wl, int dim)
        {
            if (wl.Count == 1)
            {
                return wl[0];
            }

            List<string> wlNameList = new List<string>();
            List<Tensor> twl = new List<Tensor>();
            long sumDimSize = 0;
            bool needGradient = false;

            foreach (IWeightTensor item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sumDimSize += m.Sizes[dim];

                twl.Add(m.TWeight);
                wlNameList.Add(item.Name);

                needGradient = (needGradient || m.NeedGradient);
            }

            long[] newSizes = new long[wl[0].Sizes.Length];
            for (int i = 0; i < newSizes.Length; i++)
            {
                newSizes[i] = wl[0].Sizes[i];
            }
            newSizes[dim] = sumDimSize;


            string wlName = string.Join("_", wlNameList);
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(newSizes, m_deviceId, name: $"{GetHashString(wlName)}.Concat", graphToBind: this, needGradient: needGradient);
            VisualizeNodes(wl, res);

            Ops.Concat(res.TWeight, dim, twl.ToArray());

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.ReleaseWeight();
                    bool isOwnerExclusive = res.TGradient.IsOwnerExclusive();

                    long sx = 0;
                    foreach (IWeightTensor item in wl)
                    {
                        WeightTensor m = item as WeightTensor;
                        if (item.NeedGradient)
                        {
                            using Tensor tTmp = res.TGradient.Narrow(dim, sx, m.Sizes[dim]);
                            if (isOwnerExclusive && m.IsGradientNull())
                            {
                                m.TGradient = tTmp.CopyRef();
                            }
                            else
                            {
                                m.CopyOrAddGradient(tTmp, res.Name);
                            }
                        }

                        sx += m.Sizes[dim];
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor TransposeBatch(IWeightTensor m, int batchSize)
        {
            WeightTensor t = m as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(t.Sizes, m_deviceId, name: $"{GetHashString(m.Name)}.TransposeBatch", graphToBind: this, needGradient: t.NeedGradient);
            VisualizeNodes(m, res);

            int sizeEveryBatch = m.Rows / batchSize;
            using (Tensor tWView = t.TWeight.View(sizeEveryBatch, batchSize, m.Columns))
            {
                using Tensor tWViewPermute = tWView.Permute(1, 0, 2);
                using Tensor tW2 = Ops.AsContiguous(tWViewPermute);
                res.TWeight = tW2.View(m.Rows, m.Columns);
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (t.NeedGradient)
                    {
                        res.ReleaseWeight();
                        using Tensor g = t.TGradient.View(sizeEveryBatch, batchSize, m.Columns);
                        using Tensor t2 = res.TGradient.View(batchSize, sizeEveryBatch, m.Columns);
                        using Tensor t2Permute = t2.Permute(1, 0, 2);
                        Ops.Add(g, g, t2Permute);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }
       
        public List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes)
        {
            WeightTensor m = w as WeightTensor;
            List<IWeightTensor> resList = new List<IWeightTensor>();

            int x = 0;
            foreach (int size in sizes)
            {
                WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Rows, size, m_deviceId, name: $"{GetHashString(w.Name)}.SplitColumn", graphToBind: this, needGradient: m.NeedGradient);
                VisualizeNodes(w, res);

                res.TWeight = m.TWeight.Narrow(1, x, size);
                resList.Add(res);

                x += size;
            }


            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        x = 0;
                        int i = 0;
                        foreach (IWeightTensor item in resList)
                        {
                            WeightTensor item_i = item as WeightTensor;
                            using (Tensor mG = m.TGradient.Narrow(1, x, sizes[i]))
                            {
                                Ops.Add(mG, mG, item_i.TGradient);
                            }

                            item.Dispose();

                            x += sizes[i];
                            i++;
                        }
                    }
                    else
                    {
                        foreach (IWeightTensor item in resList)
                        {
                            item.Dispose();
                        }
                    }
                }
                m_backprop.Add(backward);
            }


            return resList;
        }

        public IWeightTensor AsContiguous(IWeightTensor w, bool shareTensor = true)
        {
            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.AsContiguous", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            res.TWeight = Ops.AsContiguous(m.TWeight);

            if (shareTensor)
            {
                m.ReleaseWeight();
                m.TWeight = res.TWeight.CopyRef();
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        if (res.TGradient.IsOwnerExclusive() && m.IsGradientNull())
                        {
                            m.TGradient = res.TGradient.CopyRef();
                        }
                        else
                        {
                            m.CopyOrAddGradient(res);
                        }
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;


        }

        public IWeightTensor View(IWeightTensor w, params long[] dims)
        {
            bool hasNegOne = false;
            int negOneIdx = 0;
            long totalGivenSize = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                long dim = dims[i];
                if (dim == -1)
                {
                    if (hasNegOne)
                    {
                        throw new ArgumentException($"View operation only allows single -1 in dims.");
                    }

                    hasNegOne = true;
                    negOneIdx = i;
                }
                else
                {
                    totalGivenSize *= dim;
                }
            }

            if (hasNegOne)
            {
                long totalSrcSize = 1;
                foreach (int size in w.Sizes)
                {
                    totalSrcSize *= size;
                }

                dims[negOneIdx] = totalSrcSize / totalGivenSize;
            }


            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(dims, m_deviceId, name: $"{w.Name}.View", graphToBind: this, needGradient: m.NeedGradient);
            //  VisualizeNodes(w, res);

            res.TWeight = m.TWeight.View(dims);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();
                        bool isOwnerExclusive = res.TGradient.IsOwnerExclusive();

                        using Tensor resGConti = Ops.AsContiguous(res.TGradient);
                        using Tensor resG = resGConti.View(m.Sizes);
                        if (isOwnerExclusive && m.IsGradientNull())
                        {
                            m.TGradient = resG.CopyRef();
                        }
                        else
                        {
                            m.CopyOrAddGradient(resG, res.Name);
                        }
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Scatter(IWeightTensor source, IWeightTensor indices, int dim, params long[] shape)
        {
            WeightTensor s = source as WeightTensor;
            WeightTensor i = indices as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(shape, m_deviceId, name: $"{GetHashString(s.Name + i.Name)}.Scatter", graphToBind: this, needGradient: s.NeedGradient);

            Ops.Fill(res.TWeight, 0.0f);
            Ops.Scatter(res.TWeight, s.TWeight, dim, i.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (s.NeedGradient)
                    {
                        res.ReleaseWeight();
                        using var tmp = Ops.Gather(null, res.TGradient, dim, i.TWeight);
                        s.CopyOrAddGradient(tmp);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor ScatterAdd(IWeightTensor source, IWeightTensor indices, int dim, params long[] shape)
        {
            WeightTensor s = source as WeightTensor;
            WeightTensor i = indices as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(shape, m_deviceId, name: $"{GetHashString(s.Name + i.Name)}.Scatter", graphToBind: this, needGradient: s.NeedGradient);

            Ops.Fill(res.TWeight, 0.0f);
            Ops.ScatterAdd(res.TWeight, s.TWeight, dim, i.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (s.NeedGradient)
                    {
                        res.ReleaseWeight();
                        using var tmp = Ops.Gather(null, res.TGradient, dim, i.TWeight);
                        s.CopyOrAddGradient(tmp);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Scatter(IWeightTensor indices, float val, int dim, bool needGradient = true, params long[] shape)
        {
            WeightTensor i = indices as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(shape, m_deviceId, name: $"{GetHashString(i.Name)}.Scatter", graphToBind: this, needGradient: needGradient);

            Ops.Fill(res.TWeight, 0.0f);
            Ops.ScatterFill(res.TWeight, val, dim, i.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (needGradient)
                    {
                        res.ReleaseWeight();
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Expand(IWeightTensor w, params long[] dims)
        {

            WeightTensor m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(dims, m_deviceId, name: $"{GetHashString(w.Name)}.Expand", graphToBind: this, needGradient: m.NeedGradient);
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Expand(dims);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (m.NeedGradient)
                    {
                        res.ReleaseWeight();

                        using var tmpMGrad = m.TGradient.Expand(dims); // expand input tensor at first
                        Ops.AtomicAdd(tmpMGrad, res.TGradient);


                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2)
        {
            List<IWeightTensor> res = SplitColumns2(w, size1, size2);

            return (res[0], res[1]);
        }

        public (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3)
        {
            List<IWeightTensor> res = SplitColumns2(w, size1, size2, size3);

            return (res[0], res[1], res[2]);
        }

        private Tensor BuildRandomTensor(int rows, int columns, int batchSize, float prob)
        {
            using Tensor noise = new Tensor(TensorAllocator.Allocator(m_deviceId), DType.Float32, rows / batchSize, columns);
            float[] w = TensorSharp.RandomGenerator.BuildRandomBernoulliWeight(new long[] { rows / batchSize, columns }, prob);
            noise.SetElementsAsFloat(w);

            if (rows / batchSize == 1)
            {
                return noise.Expand(rows, columns);
            }
            else
            {
                return noise.RepeatTensor(batchSize, 1);
            }
        }

        public IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-9f)
        {
            WeightTensor srcT = src as WeightTensor;
            WeightTensor alphaT = alpha as WeightTensor;
            WeightTensor betaT = beta as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(srcT.Sizes, m_deviceId, name: $"{GetHashString(src.Name, alpha.Name, beta.Name)}.LayerNorm", graphToBind: this, needGradient: srcT.NeedGradient);
            VisualizeNodes(new IWeightTensor[] { src, alpha, beta }, res);

            Ops.LayerNorm(res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);
            if (m_needsBackprop)
            {
                void backward()
                {
                    if (srcT.NeedGradient)
                    {
                        Ops.LayerNormGrad(srcT.TGradient, alphaT.TGradient, betaT.TGradient, res.TGradient, res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);

                srcT.UnbindFromComputeGraph();
                alphaT.UnbindFromComputeGraph();
                betaT.UnbindFromComputeGraph();
            }

            return res;
        }



        ///// <summary>
        ///// LayerNorm (src1 + src2)
        ///// </summary>
        ///// <param name="src1"></param>
        ///// <param name="src2"></param>
        ///// <param name="alpha"></param>
        ///// <param name="beta"></param>
        ///// <param name="eps"></param>
        ///// <returns></returns>
        //public IWeightTensor AddLayerNorm(IWeightTensor src1, IWeightTensor src2, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f)
        //{
        //    WeightTensor src1T = src1 as WeightTensor;
        //    WeightTensor src2T = src2 as WeightTensor;
        //    WeightTensor alphaT = alpha as WeightTensor;
        //    WeightTensor betaT = beta as WeightTensor;

        //    WeightTensor res = m_weightTensorFactory.CreateWeightTensor(src1T.Sizes, m_deviceId, name: $"{GetHashString(src1.Name, src2.Name, alpha.Name, beta.Name)}.AddLayerNorm");
        //    VisualizeNodes(new IWeightTensor[] { src1, src2, alpha, beta }, res);

        //    Ops.AddLayerNorm(res.TWeight, src1T.TWeight, src2T.TWeight, alphaT.TWeight, betaT.TWeight, eps);
        //    if (m_needsBackprop)
        //    {
        //        void backward()
        //        {
        //            Ops.AddLayerNormGrad(src1T.TGradient, src2T.TGradient, alphaT.TGradient, betaT.TGradient, res.TGradient, res.TWeight, src1T.TWeight, src2T.TWeight, alphaT.TWeight, betaT.TWeight, eps);

        //            res.Dispose();
        //        }
        //        m_backprop.Add(backward);

        //        src1T.UnbindFromComputeGraph();
        //        src2T.UnbindFromComputeGraph();

        //        alphaT.UnbindFromComputeGraph();
        //        betaT.UnbindFromComputeGraph();
        //    }

        //    return res;
        //}


        public IWeightTensor Dropout(IWeightTensor V, int batchSize, float drop_prob, bool inPlace = false)
        {
            if (drop_prob == 0 || !m_needsBackprop)
            {
                return V;
            }

            // Generate noise tensor
            float p = 1.0f - drop_prob;
            Tensor noise = BuildRandomTensor(V.Rows, V.Columns, batchSize, p);

            WeightTensor w = V as WeightTensor;
            WeightTensor res = null;
            if (inPlace)
            {
                res = w.CopyWeightsRef($"{GetHashString(V.Name)}.Dropout", needGradient: w.NeedGradient);
            }
            else
            {
                res = m_weightTensorFactory.CreateWeightTensor(w.Sizes, m_deviceId, name: $"{GetHashString(V.Name)}.Dropout", graphToBind: this, needGradient: w.NeedGradient);
            }
            VisualizeNodes(V, res);

            Ops.Mul(res.TWeight, w.TWeight, noise);

            void backward()
            {
                if (w.NeedGradient)
                {
                    res.ReleaseWeight();

                    if (inPlace && w.IsGradientNull() && res.TGradient.IsOwnerExclusive())
                    {
                        w.TGradient = res.TGradient.CopyRef();
                    }

                    w.AddMulGradient(noise, res.TGradient, inPlace);
                }

                res.Dispose();
                noise.Dispose();
            }
            m_backprop.Add(backward);


            return res;
        }


        public IWeightTensor Gather(IWeightTensor src, IWeightTensor indices, int dim)
        {
            WeightTensor i = indices as WeightTensor;
            WeightTensor s = src as WeightTensor;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(indices.Sizes, m_deviceId, name: $"Gather_{m_deviceId}", graphToBind: this, needGradient: s.NeedGradient);
            Ops.Gather(res.TWeight, s.TWeight, dim, i.TWeight);

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (s.NeedGradient)
                    {
                        res.ReleaseWeight();
                        Ops.ScatterAdd(s.TGradient, res.TGradient, dim, i.TWeight);
                    }

                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Select(IWeightTensor src, int dim, int index)
        {
            WeightTensor s = src as WeightTensor;
            var resTWeight = s.TWeight.Select(dim, index);

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(resTWeight.Sizes, m_deviceId, name: $"Select_{m_deviceId}", graphToBind: this, needGradient: s.NeedGradient);
            res.TWeight = resTWeight;

            if (m_needsBackprop)
            {
                void backward()
                {
                    if (s.NeedGradient)
                    {
                        res.ReleaseWeight();
                        using var tmpG = s.TGradient.Select(dim, index);
                        Ops.Add(tmpG, tmpG, res.TGradient);
                    }
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }


            return res;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input"></param>
        /// <param name="lastTokenToPad"></param>
        /// <returns>Shape: [batch_size, seq_len]</returns>
        public IWeightTensor LeftShiftTokens(List<List<int>> input, int lastTokenToPad)
        {
            float[] buf = new float[input.Count * input[0].Count];

            for (int i = 0; i < input.Count; i++)
            {
                for (int j = 0; j < input[i].Count - 1; j++)
                {
                    buf[i * input[i].Count + j] = input[i][j + 1];
                }

                buf[(i + 1) * input[i].Count - 1] = lastTokenToPad;
            }

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(input.Count, input[0].Count, m_deviceId, name: $"LeftShiftTokens_{m_deviceId}", graphToBind: this, needGradient: false);
            res.SetWeightArray(buf);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor CreateTokensTensor(List<List<int>> input)
        {
            float[] buf = new float[input.Count * input[0].Count];

            for (int i = 0; i < input.Count; i++)
            {
                for (int j = 0; j < input[i].Count; j++)
                {
                    buf[i * input[i].Count + j] = input[i][j];
                }
            }

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(input.Count, input[0].Count, m_deviceId, name: $"TokensTensor_{m_deviceId}", graphToBind: this, needGradient: false);
            res.SetWeightArray(buf);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="paddedLength"></param>
        /// <param name="appliedLengths"></param>
        /// <returns>shape: (batch_size, sequence_padded_length, dim)</returns>
        public IWeightTensor BuildFeatureMask(int paddedLength, List<int> appliedLengths, int dim)
        {
            float[] buf = new float[appliedLengths.Count * paddedLength * dim];
            Array.Fill(buf, 0.0f);

            for (int k = 0; k < appliedLengths.Count; k++)
            {
                for (int i = 0; i < appliedLengths[k]; i++)
                {
                    Array.Fill(buf, 1.0f, k * (paddedLength * dim) + i * dim, dim);
                }
            }

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { appliedLengths.Count, paddedLength, dim }, m_deviceId, name: $"FeatureMask_{m_deviceId}", graphToBind: this, needGradient: false);
            res.SetWeightArray(buf);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor BuildPadSelfMask(int paddedLength, float[] originalLengths)
        {
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { originalLengths.Length, paddedLength, paddedLength }, m_deviceId, name: $"SelfMask_{m_deviceId}", graphToBind: this, needGradient: false);
            using (Tensor originalLengthsTensor = new Tensor(res.Allocator, DType.Float32, originalLengths.Length))
            {
                originalLengthsTensor.CopyFrom(originalLengths);
                Ops.BuildSelfMask(res.TWeight, originalLengthsTensor, paddedLength, 0.0f, -99999999.0f);
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor BuildSelfTriMask(int paddedLength, float[] originalLengths)
        {
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { originalLengths.Length, paddedLength, paddedLength }, m_deviceId, name: $"SelfTriMask_{m_deviceId}", graphToBind: this, needGradient: false);
            using (Tensor originalLengthsTensor = new Tensor(res.Allocator, DType.Float32, originalLengths.Length))
            {
                originalLengthsTensor.CopyFrom(originalLengths);
                Ops.BuildSelfTriMask(res.TWeight, originalLengthsTensor, paddedLength, 0.0f, -99999999.0f);
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor BuildTriMask(int paddedLength, int batchSize)
        {
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { paddedLength, paddedLength }, m_deviceId, name: $"SelfTriMask2_{m_deviceId}", graphToBind: this, needGradient: false);
            Ops.BuildTriMask(res.TWeight, 0.0f, -99999999.0f);

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }
       
        public IWeightTensor BuildSrcTgtMask(int srcPaddedLength, int tgtPaddedLength, float[] tgtOriginalLengths, float[] srcOriginalLengths)
        {
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(new long[] { tgtOriginalLengths.Length, tgtPaddedLength, srcPaddedLength }, m_deviceId, name: $"SrcTgtMask_{m_deviceId}", graphToBind: this, needGradient: false);

            using (Tensor tgtOriginalLengthsTensor = new Tensor(res.Allocator, DType.Float32, tgtOriginalLengths.Length))
            {
                using (Tensor srcOriginalLengthsTensor = new Tensor(res.Allocator, DType.Float32, srcOriginalLengths.Length))
                {
                    srcOriginalLengthsTensor.CopyFrom(srcOriginalLengths);
                    tgtOriginalLengthsTensor.CopyFrom(tgtOriginalLengths);
                    Ops.BuildSrcTgtMask(res.TWeight, srcOriginalLengthsTensor, tgtOriginalLengthsTensor, srcPaddedLength, tgtPaddedLength, 0.0f, -99999999.0f);
                }
            }

            if (m_needsBackprop)
            {
                void backward()
                {
                    res.Dispose();
                }
                m_backprop.Add(backward);
            }

            return res;
        }


        public float CrossEntropyLoss(IWeightTensor probs, IWeightTensor truthTgtSeqs, float graident = 1.0f, float smooth = 0.0f)
        {
            var scatterIdxTensor = View(truthTgtSeqs, new long[] { -1, 1 });
            var loss = Gather(probs, scatterIdxTensor, 1);

            if (smooth > 0.0f)
            {
                loss = Add(loss, smooth);
            }

            loss = Log(loss);
            loss = Mul(loss, -1.0f);
            loss.FillGradient(graident);

            return loss.ToWeightArray().Sum() / loss.ElementCount;
        }


        public void Dispose()
        {
            // We only dispose root computing graph, For sub graph, we don't do it.
            if (m_isSubGraph == false)
            {
                if (m_backprop != null)
                {
                    m_backprop.Clear();
                }

                if (m_weightTensorFactory != null)
                {
                    m_weightTensorFactory.Dispose();
                }
            }
            else
            {
                foreach (WeightTensor item in m_tensorsBindToCurrentGraph)
                {
                    item.ReleaseWeight();
                }
            }

            m_tensorsBindToCurrentGraph.Clear();
        }
    }
}
