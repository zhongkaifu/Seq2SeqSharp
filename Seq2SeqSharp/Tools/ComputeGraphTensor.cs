using Microsoft.Msagl.Drawing;
using Microsoft.Msagl.Layout.Incremental;
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
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
        const int MaxSize = 1024000;
        T[] array;
        int count = 0;
        public int Count => count;

        public T this[int key]
        {
            get
            {
                return array[key];
            }
            set
            {
                array[key] = value;
            }
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
    }

    public class ComputeGraphTensor : IComputeGraph
    {
        WeightTensorFactory m_weightTensorFactory;
        ConcurrentList<Action> m_backprop;
        bool m_needsBackprop;
        bool m_visNeuralNetwork;
        int m_deviceId;

        // Visualization for neural network
        Microsoft.Msagl.Drawing.Graph m_opsViz;
        HashSet<string> m_setEdges;
        Microsoft.Msagl.Drawing.Subgraph m_subGraph = null;
        Dictionary<string, Microsoft.Msagl.Drawing.Subgraph> name2SubGraph = new Dictionary<string, Subgraph>();

        public ComputeGraphTensor(IWeightFactory weightFactory, int deviceId, bool needBack = true, bool visNetwork = false, ConcurrentList<Action> backprop = null)
        {
            m_backprop = backprop != null ? backprop : new ConcurrentList<Action>();
            m_weightTensorFactory = weightFactory as WeightTensorFactory;
            m_needsBackprop = needBack;
            m_deviceId = deviceId;
            m_visNeuralNetwork = visNetwork;

            if (m_visNeuralNetwork)
            {
                // Initialize parameters for neural network visualization
                m_opsViz = new Microsoft.Msagl.Drawing.Graph();
                m_setEdges = new HashSet<string>();
            }
        }

        public IComputeGraph CreateSubGraph(string name)
        {
            ComputeGraphTensor subGraph = new ComputeGraphTensor(m_weightTensorFactory, m_deviceId, m_needsBackprop, m_visNeuralNetwork, m_backprop);
            if (m_visNeuralNetwork)
            {
                // Create parameters for neural network visualization
                subGraph.m_opsViz = m_opsViz;
                subGraph.m_setEdges = m_setEdges;
                subGraph.name2SubGraph = name2SubGraph;
                if (name2SubGraph.ContainsKey(name) == false)
                {
                    int index = name.LastIndexOf(".");
                    subGraph.m_subGraph = new Subgraph(name);
                    subGraph.m_subGraph.LabelText = name.Substring(index + 1);

                    name2SubGraph.Add(name, subGraph.m_subGraph);

                    if (m_subGraph == null)
                    {
                        m_opsViz.RootSubgraph.AddSubgraph(subGraph.m_subGraph);
                    }
                    else
                    {
                        m_subGraph.AddSubgraph(subGraph.m_subGraph);
                    }
                }
                else
                {
                    subGraph.m_subGraph = name2SubGraph[name];
                }
            }

            return subGraph;
        }

        public void Backward()
        {
            for (var i = this.m_backprop.Count - 1; i >= 0; i--)
            {
                this.m_backprop[i](); // tick!
            }
        }

        public void RunTopBackward()
        {
            m_backprop[m_backprop.Count - 1]();

            m_backprop.RemoveLastItem();

        }

        public IWeightTensor BuildPositionMatrix(int row, int column)
        {
            var res = m_weightTensorFactory.BuildPositionWeightTensor(row, column, m_deviceId, name: "PositionEmbeddings", isTrainable: true);

            return res;
        }

        public IWeightTensor Sigmoid(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Sigmoid");
            VisualizeNodes(w, res);
            Ops.Sigmoid(res.TWeight, m.TWeight);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m.AddSigmoidGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
      
            
        public IWeightTensor AddTanh(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.AddTanh");
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.AddTanh(res.TWeight, m1.TWeight, m2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m1.AddTanhGradient(res);
                    m2.AddTanhGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor Mul(IWeightTensor w, float v)
        {
            var m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.MulV");
            VisualizeNodes(w, res);

            Ops.Mul(res.TWeight, m.TWeight, v);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    Ops.AddMulV(m.TGradient, m.TGradient, res.TGradient, v);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
		

        public IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var m4 = w4 as WeightTensor;

            var res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name, w3.Name, w4.Name)}.EltMulMulAdd");
            VisualizeNodes(new IWeightTensor[] { w1, w2, w3, w4 }, res);

            Ops.MulMulAdd(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    m3.AddMulGradient(m4.TWeight, res.TGradient);
                    m4.AddMulGradient(m3.TWeight, res.TGradient);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
       
        public IWeightTensor EltMul(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.EltMul");
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.Mul(res.TWeight, m1.TWeight, m2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Add(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;          
            var res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.Add");
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.Add(res.TWeight, m1.TWeight, m2.TWeight);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.CopyOrAddGradient(res);
                    m2.CopyOrAddGradient(res);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Tanh(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Tanh");
            VisualizeNodes(w, res);

            Ops.Tanh(res.TWeight, m.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    m.AddTanhGradient(res);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor Relu(IWeightTensor w)
        {
            var m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Relu");
            VisualizeNodes(w, res);

            Ops.Relu(res.TWeight, m.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    Ops.AddReluD(m.TGradient, m.TGradient, m.TWeight, res.TGradient);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }
            return res;
        }


        public IWeightTensor MulBatch(IWeightTensor m1, IWeightTensor m2, int batchSize, float alpha = 1.0f)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor((int)(batchSize * t1.TWeight.Sizes[1]), (int)t2.TWeight.Sizes[2], m_deviceId, name: $"{GetHashString(m1.Name, m2.Name)}.MulBatch");
            VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            Tensor t1W = t1.TWeight;
            Tensor t2W = t2.TWeight;
            using (Tensor rW = res.TWeight.View(batchSize, t1.TWeight.Sizes[1], t2.TWeight.Sizes[2]))
            {
                Ops.AddmmBatch(rW, 0.0f, rW, alpha, t1W, t2W);
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();
                    using (Tensor rG = res.TGradient.View(batchSize, t1.TWeight.Sizes[1], t2.TWeight.Sizes[2]))
                    {
                        using (Tensor t1G = t1.TGradient.View(t1.TWeight.Sizes[0], t1.TWeight.Sizes[1], t1.TWeight.Sizes[2]))
                        {
                            using (var tW2 = t2W.Transpose(1, 2))
                            {
                                Ops.AddmmBatch(t1G, 1.0f, t1G, 1.0f, rG, tW2);
                            }
                        }
                        using (Tensor t2G = t2.TGradient.View(t2.TWeight.Sizes[0], t2.TWeight.Sizes[1], t2.TWeight.Sizes[2]))
                        {
                            using (var tW1 = t1W.Transpose(1, 2))
                            {
                                Ops.AddmmBatch(t2G, 1.0f, t2G, 1.0f, tW1, rG);
                            }
                        }
                    }
                              
                    res.Dispose();

                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Mul(IWeightTensor m1, IWeightTensor m2)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res;

            res = m_weightTensorFactory.CreateWeightTensor(n, d, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name)}.Mul");
            VisualizeNodes(new IWeightTensor[] { m1, m2 }, res);

            Ops.Addmm(res.TWeight, 0.0f, res.TWeight, 1.0f, t1.TWeight, t2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var tW2 = t2.TWeight.Transpose())
                    {
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);
                    }

                    using (var tW1 = t1.TWeight.Transpose())
                    {
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Affine(IWeightTensor m1, IWeightTensor m2, IWeightTensor mbias)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = mbias as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(n, d, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name, mbias.Name)}.Affine");
            VisualizeNodes(new IWeightTensor[] { m1, m2, mbias }, res);

            using (var t3WExp = t3.TWeight.Expand(n, d))
            {
                Ops.Addmm(res.TWeight, 1.0f, t3WExp, 1.0f, t1.TWeight, t2.TWeight);
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var t3G = t3.TGradient.Expand(n, d))
                    {
                        Ops.Add(t3G, t3G, res.TGradient);
                    }

                    using (var tW2 = t2.TWeight.Transpose())
                    {
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);
                    }

                    using (var tW1 = t1.TWeight.Transpose())
                    {
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;

        }

        public IWeightTensor MulAdd(IWeightTensor m1, IWeightTensor m2, IWeightTensor m3)
        {            
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = m3 as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;

            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(n, d, m_deviceId, name: $"{GetHashString(m1.Name, m2.Name, m3.Name)}.MulAdd");
            VisualizeNodes(new IWeightTensor[] { m1, m2, m3 }, res);

            Ops.Addmm(res.TWeight, 1.0f, t3.TWeight, 1.0f, t1.TWeight, t2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();
                    t3.CopyOrAddGradient(res);

                    using (var tW2 = t2.TWeight.Transpose())
                    {
                        Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);
                    }

                    using (var tW1 = t1.TWeight.Transpose())
                    {
                        Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor Transpose(IWeightTensor w)
        {
            WeightTensor m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Transpose");
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.Transpose();
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    using (var gT = res.TGradient.Transpose())
                    {
                        m.CopyOrAddGradient(gT);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
     
        public IWeightTensor Softmax(IWeightTensor w, bool bp = true)
        {
            WeightTensor m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Softmax");
            VisualizeNodes(w, res);

            Ops.Softmax(res.TWeight, m.TWeight);
            if (this.m_needsBackprop && bp)
            {
                Action backward = () =>
                {
                    m.AddSoftmaxGradient(res);
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }

        public IWeightTensor PeekRow(IWeightTensor w, int ix, int num = 1, bool runGradients = true)
        {
            WeightTensor m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(num, m.Columns, m_deviceId, name: $"{GetHashString(w.Name)}.PeekRow");
            res.TWeight = m.TWeight.Narrow(0, ix, num);
            res.TGradient = (m.TGradient != null && runGradients) ? m.TGradient.Narrow(0, ix, num) : null;

            VisualizeNodes(w, res);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
    
        public IWeightTensor ConcatColumns(IWeightTensor w1, IWeightTensor w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;

            int sx = m1.Rows;
            int sy = m1.Columns + m2.Columns;

            var res = m_weightTensorFactory.CreateWeightTensor(sx, sy, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name)}.ConcatColumns");
            VisualizeNodes(new IWeightTensor[] { w1, w2 }, res);

            Ops.Concat(res.TWeight, 1, m1.TWeight, m2.TWeight);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (Tensor tTmp1 = res.TGradient.Narrow(1, 0, m1.Columns))
                    {
                        m1.CopyOrAddGradient(tTmp1);
                    }

                    using (Tensor tTmp2 = res.TGradient.Narrow(1, m1.Columns, m2.Columns))
                    {
                        m2.CopyOrAddGradient(tTmp2);
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }
            return res;
        }

        private byte[] GetHash(string inputString)
        {
            HashAlgorithm algorithm = SHA256.Create();
            return algorithm.ComputeHash(Encoding.UTF8.GetBytes(inputString));
        }

        private string GetHashString(params string[] inputStrings)
        {
            if (m_visNeuralNetwork)
            {
                string inputString = String.Join("_", inputStrings);
                StringBuilder sb = new StringBuilder();
                foreach (byte b in GetHash(inputString))
                    sb.Append(b.ToString("X2"));

                return sb.ToString();
            }
            return String.Empty;
        }
      
        private void VisualizeNodes(IWeightTensor sourceNode, IWeightTensor targetNode)
        {
            VisualizeNodes(new IWeightTensor[] { sourceNode }, targetNode);
        }

        private void VisualizeNodes(IEnumerable<IWeightTensor> sourceNodes, IWeightTensor targetNode)
        {
            if (!m_visNeuralNetwork || m_deviceId != 0)
            {
                return;
            }

            // Create node for target tensor
            int index = targetNode.Name.LastIndexOf('.');
            Microsoft.Msagl.Drawing.Node tgtNode = m_opsViz.AddNode(targetNode.Name);
            tgtNode.LabelText = targetNode.Name.Substring(index + 1);

            if (targetNode.IsTrainable)
            {
                tgtNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;
            }

            if (m_subGraph != null)
            {
                // Current compute graph is a sub-graph
                m_subGraph.AddNode(tgtNode);
            }

            // Create edges for each source node and target node
            foreach (var sourceNode in sourceNodes)
            {
                if (!String.IsNullOrEmpty(sourceNode.Name) && !String.IsNullOrEmpty(targetNode.Name))
                {
                    string key = $"{sourceNode.Name}->{targetNode.Name}";
                    if (m_setEdges.Contains(key))
                    {
                        continue;
                    }

                    int srcIndex = sourceNode.Name.LastIndexOf('.');
                    Microsoft.Msagl.Drawing.Node srcNode = m_opsViz.AddNode(sourceNode.Name);
                    srcNode.LabelText = sourceNode.Name.Substring(srcIndex + 1);
                    if (sourceNode.IsTrainable)
                    {
                        srcNode.Attr.FillColor = Microsoft.Msagl.Drawing.Color.LightSteelBlue;

                        if (m_subGraph != null)
                        {
                            m_subGraph.AddNode(srcNode);
                        }
                    }

                    var edge = m_opsViz.AddEdge(sourceNode.Name, targetNode.Name);

                    m_setEdges.Add(key);
                }
            }
        }

        public void VisualizeNeuralNetToFile(string neuralNetPicFilePath)
        {
            var settings = new FastIncrementalLayoutSettings();
            settings.AvoidOverlaps = true;
            settings.NodeSeparation = 30;
            settings.RouteEdges = true;

            m_opsViz.LayoutAlgorithmSettings = settings;

            var renderer = new Microsoft.Msagl.GraphViewerGdi.GraphRenderer(m_opsViz);
            renderer.CalculateLayout();

            var bitmap = new System.Drawing.Bitmap((int)m_opsViz.Width, (int)m_opsViz.Height, System.Drawing.Imaging.PixelFormat.Format32bppPArgb);
            renderer.Render(bitmap);

            bitmap.Save(neuralNetPicFilePath);

            bitmap.Dispose();
        }
      
        public IWeightTensor RepeatRows(IWeightTensor w, int n)
        {
            var m = w as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(m.Rows * n, m.Columns, m_deviceId, name: $"{GetHashString(w.Name)}.RepeatRows");
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.RepeatTensor(n, 1);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    for (int i = 0; i < n; i++)
                    {
                        using (var resG_i = res.TGradient.Narrow(0, m.Rows * i, m.Rows))
                        {
                            m.CopyOrAddGradient(resG_i);
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }


            return res;
        }


        public IWeightTensor ConcatRows(List<IWeightTensor> wl)
        {
            if (wl.Count == 1)
            {
                return wl[0];
            }

            List<string> wlNameList = new List<string>();
            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;
            foreach (IWeightTensor item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx += m.Rows;
                sy = m.Columns;

                twl.Add(m.TWeight);
                wlNameList.Add(item.Name);
            }

            var wlName = String.Join("_", wlNameList);
            var res = m_weightTensorFactory.CreateWeightTensor(sx, sy, m_deviceId, name: $"{GetHashString(wlName)}.ConcatRows");
            VisualizeNodes(wl, res);

            Ops.Concat(res.TWeight, 0, twl.ToArray());

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sx = 0;
                    foreach (IWeightTensor item in wl)
                    {
                        WeightTensor m = item as WeightTensor;
                        using (var tTmp = res.TGradient.Narrow(0, sx, m.Rows))
                        {
                            m.CopyOrAddGradient(tTmp);
                            sx += m.Rows;
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }
            return res;
        }
      
        public IWeightTensor TransposeBatch(IWeightTensor m, int batchSize)
        {
            WeightTensor t = m as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(t.Sizes, m_deviceId, name: $"{GetHashString(m.Name)}.TransposeBatch");
            VisualizeNodes(m, res);

            int sizeEveryBatch = m.Rows / batchSize;
            using (var tWView = t.TWeight.View(sizeEveryBatch, batchSize, m.Columns))
            {
                using (var tWViewPermute = tWView.Permute(1, 0, 2))
                {
                    using (var tW2 = Ops.AsContiguous(tWViewPermute))
                    {
                        res.TWeight = tW2.View(m.Rows, m.Columns);
                    }
                }
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    using (var g = t.TGradient.View(sizeEveryBatch, batchSize, m.Columns))
                    {
                        using (var t2 = res.TGradient.View(batchSize, sizeEveryBatch, m.Columns))
                        {
                            using (var t2Permute = t2.Permute(1, 0, 2))
                            {
                                Ops.Add(g, g, t2Permute);
                            }
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }


            return res;
        }

        public IWeightTensor ConcatColumns(params IWeightTensor[] wl)
        {
            if (wl.Length == 1)
            {
                return wl[0];
            }

            List<string> srcNameList = new List<string>();
            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightTensor item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx = m.Rows;
                sy += m.Columns;

                twl.Add(m.TWeight);
                srcNameList.Add(item.Name);
            }

            string srcNames = String.Join("_", srcNameList);
            var res = m_weightTensorFactory.CreateWeightTensor(sx, sy, m_deviceId, name: $"{GetHashString(srcNames)}.ConcatColumns");
            VisualizeNodes(wl, res);

            Ops.Concat(res.TWeight, 1, twl.ToArray());
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sy = 0;
                    foreach (IWeightTensor item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        using (Tensor tTmp = res.TGradient.Narrow(1, sy, m.Columns))
                        {
                            m.CopyOrAddGradient(tTmp);
                            sy += m.Columns;
                        }
                    }

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }
            return res;
        }

        public List<IWeightTensor> SplitColumns2(IWeightTensor w, params int[] sizes)
        {
            var m = w as WeightTensor;
            List<IWeightTensor> resList = new List<IWeightTensor>();

            int x = 0;
            foreach (int size in sizes)
            {
                WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Rows, size, m_deviceId, name: $"{GetHashString(w.Name)}.SplitColumn");
                VisualizeNodes(w, res);

                res.TWeight = m.TWeight.Narrow(1, x, size);
                resList.Add(res);

                x += size;
            }


            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    x = 0;
                    int i = 0;
                    foreach (var item in resList)
                    {
                        var item_i = item as WeightTensor;
                        using (var mG = m.TGradient.Narrow(1, x, sizes[i]))
                        {
                            Ops.Add(mG, mG, item_i.TGradient);
                        }

                        item.Dispose();

                        x += sizes[i];
                        i++;
                    }
                };
                this.m_backprop.Add(backward);
            }


            return resList;
        }

        public IWeightTensor Permute(IWeightTensor w, params int[] dims)
        {
            var m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(m.Sizes, m_deviceId, name: $"{GetHashString(w.Name)}.Permute");
            VisualizeNodes(w, res);

            using (var tWPremute = m.TWeight.Permute(dims))
            {
                res.TWeight = Ops.AsContiguous(tWPremute);
            }

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    using (var gT = m.TGradient.Permute(dims))
                    {
                        Ops.Add(gT, gT, res.TGradient);
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }


        public IWeightTensor View(IWeightTensor w, params long[] dims)
        {
            var m = w as WeightTensor;
            WeightTensor res = m_weightTensorFactory.CreateWeightTensor(dims, m_deviceId, name: $"{GetHashString(w.Name)}.View");
            VisualizeNodes(w, res);

            res.TWeight = m.TWeight.View(dims);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    using (var resG = res.TGradient.View(m.TWeight.Sizes))
                    {
                        m.CopyOrAddGradient(resG);
                    }
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }


            return res;
        }

        public (IWeightTensor r1, IWeightTensor r2) SplitColumns(IWeightTensor w, int size1, int size2)
        {
            var res = SplitColumns2(w, size1, size2);

            return (res[0], res[1]);
        }

        public (IWeightTensor r1, IWeightTensor r2, IWeightTensor r3) SplitColumns(IWeightTensor w, int size1, int size2, int size3)
        {
            var res = SplitColumns2(w, size1, size2, size3);

            return (res[0], res[1], res[2]);
        }
       
        Random rnd = new Random(DateTime.Now.Millisecond);
        private Tensor BuildRandomTensor(int rows, int columns, double prob)
        {
            float[] weights = new float[rows * columns];
            for (int i = 0; i < weights.Length; i++)
            {
                double r = rnd.NextDouble();
                if (r < prob)
                {
                    weights[i] = 1.0f;
                }
            }

            Tensor noise = new Tensor(TensorAllocator.Allocator(m_deviceId), DType.Float32, rows, columns);
            noise.SetElementsAsFloat(weights);

            return noise;
        }

        public IWeightTensor LayerNorm(IWeightTensor src, IWeightTensor alpha, IWeightTensor beta, float eps = 1e-09f)
        {
            var srcT = src as WeightTensor;
            var alphaT = alpha as WeightTensor;
            var betaT = beta as WeightTensor;

            var alphaTWExp = alphaT.TWeight.Expand(src.Rows, src.Columns);
            var betaTWExp = betaT.TWeight.Expand(src.Rows, src.Columns);

            var res = m_weightTensorFactory.CreateWeightTensor(src.Rows, src.Columns, m_deviceId, name:$"{GetHashString(src.Name, alpha.Name, beta.Name)}.LayerNorm");
            VisualizeNodes(new IWeightTensor[] { src, alpha, beta }, res);

            Ops.LayerNorm(res.TWeight, srcT.TWeight, alphaTWExp, betaTWExp, eps);
            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    using (var alphaTGExp = alphaT.TGradient.Expand(src.Rows, src.Columns))
                    {
                        using (var betaTGExp = betaT.TGradient.Expand(src.Rows, src.Columns))
                        {
                            Ops.LayerNormGrad(srcT.TGradient, alphaTGExp, betaTGExp, res.TGradient, res.TWeight, srcT.TWeight, alphaTWExp, betaTWExp, eps);
                        }
                    }

                    alphaTWExp.Dispose();
                    betaTWExp.Dispose();

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }
            else
            {
                alphaTWExp.Dispose();
                betaTWExp.Dispose();
            }

            return res;
        }

        public IWeightTensor Dropout(IWeightTensor V, float drop_prob)
        {
            float p = 1.0f - drop_prob;
            var w = V as WeightTensor;
            var res = m_weightTensorFactory.CreateWeightTensor(V.Rows, V.Columns, m_deviceId, name: $"{GetHashString(V.Name)}.Dropout");
            VisualizeNodes(V, res);

            Tensor noise = BuildRandomTensor(V.Rows, V.Columns, p);
            Ops.Mul(res.TWeight, w.TWeight, noise);

            if (this.m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();
                    w.AddMulGradient(noise, res.TGradient);

                    noise.Dispose();
                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
    }
}
