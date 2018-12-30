using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp.Tools
{
    public class ComputeGraphTensor : IComputeGraph
    {
        internal static WeightTensorFactory weightTensorFactory;
        public ConcurrentList<Action> backprop = new ConcurrentList<Action>();
        public bool needs_backprop { get; set; }

        public ComputeGraphTensor(IWeightFactory weightFactory, bool needBack = true)
        {
            weightTensorFactory = weightFactory as WeightTensorFactory;
            weightTensorFactory.Clear();

            needs_backprop = needBack;
        }

        public void Backward()
        {
            for (var i = this.backprop.Count - 1; i >= 0; i--)
            {
                this.backprop[i](); // tick!
            }
        }


        public IWeightMatrix Sigmoid(IWeightMatrix w)
        {
            var m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);
            Ops.Sigmoid(res.TWeight, m.TWeight);

            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    Ops.AddSigmoidD(m.TGradient, m.TGradient, res.TWeight, res.TGradient);
                };
                this.backprop.Add(backward);
            }

            return res;

        }

        public IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns);

            Ops.Add3(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight);

            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    Ops.Add(m1.TGradient, m1.TGradient, res.TGradient);
                    Ops.Add(m2.TGradient, m2.TGradient, res.TGradient);
                    Ops.Add(m3.TGradient, m3.TGradient, res.TGradient);
                };
                this.backprop.Add(backward);
            }

            return res;

        }

        public IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var m4 = w4 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns);
            Ops.Add4(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);

            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    Ops.Add(m1.TGradient, m1.TGradient, res.TGradient);
                    Ops.Add(m2.TGradient, m2.TGradient, res.TGradient);
                    Ops.Add(m3.TGradient, m3.TGradient, res.TGradient);
                    Ops.Add(m4.TGradient, m4.TGradient, res.TGradient);
                };
                this.backprop.Add(backward);
            }

            return res;

        }      

        public IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns);
            Ops.AddTanh(res.TWeight, m1.TWeight, m2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Tensor tTmp = Ops.TanhD(null, res.TWeight, res.TGradient);
                    Ops.Add(m1.TGradient, m1.TGradient, tTmp);
                    Ops.Add(m2.TGradient, m2.TGradient, tTmp);
                    tTmp.Dispose();


                };
                this.backprop.Add(backward);
            }

            return res;

        }

      

        public IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns);

            Ops.Mul(res.TWeight, m1.TWeight, m2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.AddMul(m1.TGradient, m1.TGradient, m2.TWeight, res.TGradient);
                    Ops.AddMul(m2.TGradient, m2.TGradient, m1.TWeight, res.TGradient);

                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;          
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns);

            Ops.Add(res.TWeight, m1.TWeight, m2.TWeight);

            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    Ops.Add(m1.TGradient, res.TGradient, m1.TGradient);
                    Ops.Add(m2.TGradient, res.TGradient, m2.TGradient);

                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public IWeightMatrix Tanh(IWeightMatrix w)
        {
            var m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);
            Ops.Tanh(res.TWeight, m.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.AddTanhD(m.TGradient, m.TGradient, res.TWeight, res.TGradient);
                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public IWeightMatrix Mul(IWeightMatrix m1, IWeightMatrix m2)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res;

            res = weightTensorFactory.CreateWeightTensor(n, d);
            Ops.Addmm(res.TWeight, 0.0f, res.TWeight, 1.0f, t1.TWeight, t2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    var tW2 = t2.TWeight.Transpose();
                    var tTmp1 = Ops.Addmm(null, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);               
                    Ops.Copy(t1.TGradient, tTmp1);


                    var tW1 = t1.TWeight.Transpose();
                    var tTmp2 = Ops.Addmm(null, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    Ops.Copy(t2.TGradient, tTmp2);


                    tW1.Dispose();
                    tW2.Dispose();

                    tTmp1.Dispose();
                    tTmp2.Dispose();

                };
                this.backprop.Add(backward);
            }

            return res;
        }


        public IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3)
        {            
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = m3 as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;

            WeightTensor res = weightTensorFactory.CreateWeightTensor(n, d);
            Ops.Addmm(res.TWeight, 1.0f, t3.TWeight, 1.0f, t1.TWeight, t2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.Add(t3.TGradient, t3.TGradient, res.TGradient);

                    var tW2 = t2.TWeight.Transpose();
                    var tTmp1 = Ops.Addmm(null, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);
                    Ops.Copy(t1.TGradient, tTmp1);


                    var tW1 = t1.TWeight.Transpose();
                    var tTmp2 = Ops.Addmm(null, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);
                    Ops.Copy(t2.TGradient, tTmp2);

                    tW1.Dispose();
                    tW2.Dispose();

                    tTmp1.Dispose();
                    tTmp2.Dispose();

                };
                this.backprop.Add(backward);
            }

            return res;
        }



        public IWeightMatrix MulAdd2(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = m3 as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;

            WeightTensor res = weightTensorFactory.CreateWeightTensor(n, d);
            Ops.Addmm(res.TWeight, 1.0f, t3.TWeight, 1.0f, t1.TWeight, t2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.Add(t3.TGradient, t3.TGradient, res.TGradient);

                    var tW2 = t2.TWeight.Transpose();
                    Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);


                    var tW1 = t1.TWeight.Transpose();
                    Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);

                    tW1.Dispose();
                    tW2.Dispose();

                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public IWeightMatrix SoftmaxWithCrossEntropy(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);

            var maxval = Ops.MaxAll(m.TWeight);
            Ops.ExpSub(res.TWeight, m.TWeight, maxval);
            float s = Ops.SumAll(res.TWeight);
            Ops.Mul(res.TWeight, res.TWeight, 1.0f / s);

            return res;

        }

      


        public IWeightMatrix Transpose2(IWeightMatrix w)
        {
            WeightTensor m = w as WeightTensor;

            var wT = m.TWeight.Transpose();
            var gT = m.TGradient.Transpose();

          //  var res = new WeightTensor(m.Columns, m.Rows, wT, gT);

            var res = weightTensorFactory.CreateWeightTensor(m.Columns, m.Rows, wT, gT);

            //wT.Dispose();
            //gT.Dispose();

            return res;
        }

        public IWeightMatrix Softmax(IWeightMatrix w)
        {
            WeightTensor m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);

            var maxval = Ops.MaxAll(m.TWeight);
            Ops.ExpSub(res.TWeight, m.TWeight, maxval);
            float s = Ops.SumAll(res.TWeight);
            Ops.Mul(res.TWeight, res.TWeight, 1.0f / s);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Tensor tTmp = Ops.Mul(null, res.TGradient, res.TWeight);
                    Ops.Add(m.TGradient, m.TGradient, tTmp);
                    float ss = Ops.SumAll(tTmp);

                    Ops.AddMulV(m.TGradient, m.TGradient, res.TWeight, -ss);

                    tTmp.Dispose();
                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public IWeightMatrix SoftmaxM(IWeightMatrix w, bool bp = true)
        {
            WeightTensor m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, new Tensor(TensorAllocator.Allocator, DType.Float32, m.Rows, m.Columns), bp);

            Tensor tTmp = new Tensor(TensorAllocator.Allocator, DType.Float32, m.Rows, m.Columns);

            var maxval = Ops.Max(null, m.TWeight, 1);
            var maxvalM = maxval.Expand(m.Rows, m.Columns);

            Ops.ExpSub2(tTmp, m.TWeight, maxvalM);

            var sumV = Ops.Sum(null, tTmp, 1);
            var sumM = sumV.Expand(m.Rows, m.Columns);
            Ops.Div(res.TWeight, tTmp, sumM);

            maxval.Dispose();
            maxvalM.Dispose();
            sumV.Dispose();
            sumM.Dispose();

            if (this.needs_backprop && bp)
            {
                Action backward = () =>
                {
                    Ops.Mul(tTmp, res.TGradient, res.TWeight);
                    Ops.Add(m.TGradient, m.TGradient, tTmp);

                    var ss = Ops.Sum(null, tTmp, 1);
                    var ssN = Ops.Neg(null, ss);

                    var ssM = ssN.Expand(m.Rows, m.Columns);
                    Ops.AddMul(m.TGradient, m.TGradient, res.TWeight, ssM);


                    tTmp.Dispose();
                    ss.Dispose();
                    ssM.Dispose();
                    ssN.Dispose();
                };
                this.backprop.Add(backward);
            }
            else
            {
                tTmp.Dispose();
            }

            return res;
        }

        public IWeightMatrix PeekRow(IWeightMatrix w, int ix)
        {
            WeightTensor m = w as WeightTensor;
            var d = m.Columns;
            var tw = m.TWeight.Narrow(0, ix, 1);
            var tg = m.TGradient != null ? m.TGradient.Narrow(0, ix, 1) : null;

            var res = weightTensorFactory.CreateWeightTensor(1, m.Columns, tw, tg);

            if (m.RowToBeUpdated.ContainsKey(ix) == false)
            {
                m.RowToBeUpdated.Add(ix, 1);
            }
            else
            {
                m.RowToBeUpdated[ix]++;
            }

            return res;
        }

        //public void DropoutPredict(IWeightMatrix V, float drop_prob)
        //{
        //    WeightTensor m = V as WeightTensor;
        //    Ops.Mul(m.TWeight, m.TWeight, 0.2f);
        //}


        public IWeightMatrix ConcatColumns(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;

            int sx = m1.Rows;
            int sy = m1.Columns + m2.Columns;

            var res = weightTensorFactory.CreateWeightTensor(sx, sy);

            Ops.Concat(res.TWeight, 1, m1.TWeight, m2.TWeight);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Tensor tTmp1 = res.TGradient.Narrow(1, 0, m1.Columns);
                    Ops.Add(m1.TGradient, m1.TGradient, tTmp1);

                    Tensor tTmp2 = res.TGradient.Narrow(1, m1.Columns, m2.Columns);
                    Ops.Add(m2.TGradient, m2.TGradient, tTmp2);

                    tTmp1.Dispose();
                    tTmp2.Dispose();
                };
                this.backprop.Add(backward);
            }
            return res;
        }

        public IWeightMatrix RepeatRows(IWeightMatrix w, int n)
        {
            var m = w as WeightTensor;
            if (m.Rows == 1)
            {
                var res = weightTensorFactory.CreateWeightTensor(m.Rows * n, m.Columns, m.TWeight.Expand(n, m.Columns), m.TGradient.Expand(n, m.Columns));

                return res;
            }
            else
            {
                List<IWeightMatrix> ws = new List<IWeightMatrix>();
                for (int i = 0; i < n; i++)
                {
                    ws.Add(w);
                }

                return ConcatRows(ws);
            }

            //var m = w as WeightTensor;
            //var res = weightTensorFactory.CreateWeightTensor(m.Rows * n, m.Columns, m.TWeight.RepeatTensor(n, 1));

            //if (this.needs_backprop)
            //{
            //    Action backward = () =>
            //    {
            //        Tensor t = Ops.Sum(null, res.TGradient, 0);
            //        Ops.Add(m.TGradient, m.TGradient, t);

            //        t.Dispose();
            //    };

            //    this.backprop.Add(backward);
            //}

            //return res;
        }


        public IWeightMatrix ConcatRows(List<IWeightMatrix> wl, bool bp = true)
        {
            if (wl.Count == 1)
            {
                return wl[0];
            }

            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightMatrix item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx += m.Rows;
                sy = m.Columns;

                twl.Add(m.TWeight);
            }


            var res = weightTensorFactory.CreateWeightTensor(sx, sy);
            Ops.Concat(res.TWeight, 0, twl.ToArray());

            if (this.needs_backprop && bp)
            {
                Action backward = () =>
                {
                    sx = 0;
                    foreach (IWeightMatrix item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        Tensor tTmp = res.TGradient.Narrow(0, sx, m.Rows);
                        Ops.Add(m.TGradient, m.TGradient, tTmp);

                        sx += m.Rows;

                        tTmp.Dispose();

                    }
                };
                this.backprop.Add(backward);
            }
            return res;

        }

        public IWeightMatrix ConcatColumns(IWeightMatrix[] wl)
        {
            if (wl.Length == 1)
            {
                return wl[0];
            }

            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightMatrix item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx = m.Rows;
                sy += m.Columns;

                twl.Add(m.TWeight);
            }


            var res = weightTensorFactory.CreateWeightTensor(sx, sy);
            Ops.Concat(res.TWeight, 1, twl.ToArray());


            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    sy = 0;
                    foreach (IWeightMatrix item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        Tensor tTmp = res.TGradient.Narrow(1, sy, m.Columns);
                        Ops.Add(m.TGradient, m.TGradient, tTmp);

                        sy += m.Columns;

                        tTmp.Dispose();
                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }

        //public List<IWeightMatrix> SplitRows(IWeightMatrix w, params int[] sizes)
        //{
        //    var m = w as WeightTensor;
        //    List<IWeightMatrix> resList = new List<IWeightMatrix>();

        //    int x = 0;
        //    foreach (int size in sizes)
        //    {
        //        Tensor tW = m.TWeight.Narrow(0, x, size);
        //        Tensor TG = m.TGradient.Narrow(0, x, size);

        //        WeightTensor res = weightTensorFactory.CreateWeightTensor(size, m.Columns, tW, TG);
        //        resList.Add(res);

        //        x += size;
        //    }

        //    return resList;
        //}

        public List<IWeightMatrix> SplitColumns(IWeightMatrix w, params int[] sizes)
        {
            var m = w as WeightTensor;
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            int x = 0;
            foreach (int size in sizes)
            {
                WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows, size, m.TWeight.Narrow(1, x, size), m.TGradient.Narrow(1, x, size));

                resList.Add(res);

                x += size;
            }

            return resList;
        }

        public List<IWeightMatrix> UnFolderRow(IWeightMatrix m, int n, bool gradient = true)
        {
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            WeightTensor t = m as WeightTensor;

            if (gradient)
            {
                Tensor tw = t.TWeight.Unfold(0, n, n);
                Tensor tG = t.TGradient.Unfold(0, n, n);

                for (int i = 0; i < n; i++)
                {
                    WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows / n, m.Columns, tw.Select(2, i), tG.Select(2, i));

                    if (res.Rows != res.TWeight.Sizes[0] || res.Rows != res.TGradient.Sizes[0])
                    {
                        throw new InvalidOperationException("Invalide unfolder");
                    }

                    resList.Add(res);
                }

                tw.Dispose();
                tG.Dispose();
            }
            else
            {
                Tensor tw = t.TWeight.Unfold(0, n, n);
                for (int i = 0; i < n; i++)
                {
                    WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows / n, m.Columns, tw.Select(2, i), gradient);

                    if (res.Rows != res.TWeight.Sizes[0])
                    {
                        throw new InvalidOperationException("Invalide unfolder");
                    }

                    resList.Add(res);
                }

                tw.Dispose();
            }

            return resList;
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

            Tensor noise = new Tensor(TensorAllocator.Allocator, DType.Float32, rows, columns);
            noise.SetElementsAsFloat(weights);

            return noise;
        }

        public IWeightMatrix Dropout(IWeightMatrix V, float drop_prob)
        {
            float p = 1.0f - drop_prob;
            var w = V as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(V.Rows, V.Columns);

            Tensor noise = BuildRandomTensor(V.Rows, V.Columns, p);
            Ops.Mul(res.TWeight, w.TWeight, noise);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.AddMul(w.TGradient, w.TGradient, res.TGradient, noise);

                    noise.Dispose();
                };
                this.backprop.Add(backward);
            }

            return res;
        }
    }
}
