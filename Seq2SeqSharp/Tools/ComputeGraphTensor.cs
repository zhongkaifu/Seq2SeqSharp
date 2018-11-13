using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp.Tools
{
    public class ComputeGraphTensor : ComputeGraphMKL
    {
        internal static WeightTensorFactory weightTensorFactory = new WeightTensorFactory();

        public ComputeGraphTensor(bool needBack = true) : base(needBack)
        {
            weightTensorFactory.Clean();
        }

      
        public override IWeightMatrix Sigmoid(IWeightMatrix w)
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

        public override IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3)
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

        public override IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4)
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

        public override IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2)
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

      

        public override IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2)
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

        public override IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2)
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

        public override IWeightMatrix Tanh(IWeightMatrix w)
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

        public override IWeightMatrix Mul(IWeightMatrix m1, IWeightMatrix m2)
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
                    Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);

                    var tW1 = t1.TWeight.Transpose();
                    Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);

                };
                this.backprop.Add(backward);
            }

            return res;
        }


        public override IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3)
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

                };
                this.backprop.Add(backward);
            }

            return res;
        }

        public override IWeightMatrix SoftmaxWithCrossEntropy(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);

            var maxval = Ops.MaxAll(m.TWeight);
            Ops.ExpSub(res.TWeight, m.TWeight, maxval);
            float s = Ops.SumAll(res.TWeight);
            Ops.Mul(res.TWeight, res.TWeight, 1.0f / s);

            return res;

        }

      


        public override IWeightMatrix Transpose2(IWeightMatrix w)
        {
            WeightTensor m = w as WeightTensor;
            var res = new WeightTensor(m.Columns, m.Rows, m.TWeight.Transpose(), m.TGradient.Transpose());

            return res;
        }

        public override IWeightMatrix Softmax(IWeightMatrix w)
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

                };
                this.backprop.Add(backward);
            }

            return res;
        }
       

        public override IWeightMatrix PeekRow(IWeightMatrix w, int ix)
        {
            WeightTensor m = w as WeightTensor;
            var d = m.Columns;
            var res = new WeightTensor(1, m.Columns, m.TWeight.Narrow(0, ix, 1), m.TGradient.Narrow(0, ix, 1));
            m.RowToBeUpdated.Add(ix);

            return res;
        }

        public override void DropoutPredict(IWeightMatrix V, float drop_prob)
        {
            WeightTensor m = V as WeightTensor;
            Ops.Mul(m.TWeight, m.TWeight, 0.2f);
        }


        public override IWeightMatrix ConcatColumns(IWeightMatrix w1, IWeightMatrix w2)
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
                    Tensor tTmp = res.TGradient.Narrow(1, 0, m1.Columns);
                    Ops.Add(m1.TGradient, m1.TGradient, tTmp);

                    tTmp = res.TGradient.Narrow(1, m1.Columns, m2.Columns);
                    Ops.Add(m2.TGradient, m2.TGradient, tTmp);                  
                };
                this.backprop.Add(backward);
            }
            return res;
        }

        public override IWeightMatrix RepeatRows(IWeightMatrix w, int n)
        {
            var m = w as WeightTensor;
            var res = new WeightTensor(m.Rows * n, m.Columns, m.TWeight.RepeatTensor(n, 1));

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    Tensor t = Ops.Sum(null, res.TGradient, 0);
                    Ops.Add(m.TGradient, m.TGradient, t);
                };

                this.backprop.Add(backward);
            }

            return res;
        }


        public override IWeightMatrix ConcatRows(List<IWeightMatrix> wl)
        {
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

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    sx = 0;
                    foreach (IWeightMatrix item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        Tensor tTmp = res.TGradient.Narrow(0, sx, m.Rows);
                        Ops.Add(m.TGradient, m.TGradient, tTmp);

                        sy += m.Rows;

                    }
                };
                this.backprop.Add(backward);
            }
            return res;

        }

        public override IWeightMatrix ConcatColumns(IWeightMatrix[] wl)
        {
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

                    }
                };
                this.backprop.Add(backward);
            }
            return res;
        }

        public override List<IWeightMatrix> SplitRows(IWeightMatrix w, params int[] sizes)
        {
            var m = w as WeightTensor;
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            int x = 0;
            foreach (int size in sizes)
            {
                Tensor tW = m.TWeight.Narrow(0, x, size);
                Tensor TG = m.TGradient.Narrow(0, x, size);

                WeightTensor res = new WeightTensor(size, m.Columns, tW, TG);
                resList.Add(res);

                x += size;
            }

            return resList;
        }

        public override List<IWeightMatrix> SplitColumns(IWeightMatrix w, params int[] sizes)
        {
            var m = w as WeightTensor;
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            int x = 0;
            foreach (int size in sizes)
            {
                Tensor tW = m.TWeight.Narrow(1, x, size);
                Tensor TG = m.TGradient.Narrow(1, x, size);

                WeightTensor res = new WeightTensor(m.Rows, size, tW, TG);
                resList.Add(res);

                x += size;
            }

            return resList;
        }

        private WeightTensor CopyWeightToTensor(WeightMatrix m)
        {
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns);

            res.TWeight = Tensor.FromArray(TensorAllocator.Allocator, m.Weight).View(m.Rows, m.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    m.Gradient = res.TGradient.GetElementsAsFloat(m.Weight.Length);
                };
                this.backprop.Add(backward);
            }
            return res;
        }

        private WeightMatrix CopyTensorToWeight(WeightTensor m)
        {
            var res = weightMatrixFactory.CreateWeightMatrix(m.Rows, m.Columns);
            res.Weight = m.TWeight.GetElementsAsFloat(m.Rows * m.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    m.TGradient.SetElementsAsFloat(res.Gradient);
                };
                this.backprop.Add(backward);
            }


            return res;
        }       
    }
}
