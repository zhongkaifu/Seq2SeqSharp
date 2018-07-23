using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    /// <summary>
    /// The matrix data storage format.
    /// </summary>
    public enum Order
    {
        /// <summary>
        /// The matrix array uses a row-major layout.
        /// </summary>
        Row = 101,

        /// <summary>
        /// The matrix array uses a column-major layout.
        /// </summary>
        Column = 102
    }

    /// <summary>
    /// Matrix transpose type.
    /// </summary>
    public enum Transpose
    {
        /// <summary>
        /// Don't transpose the matrix.  Equivalent to trans='N'
        /// </summary>
        NoTrans = 111,

        /// <summary>
        /// Transpose the matrix.  Equivalent to trans='T'
        /// </summary>
        Trans = 112,

        /// <summary>
        /// Conjugate transpose the matrix. The only refers to complex matrices. Real matrices will just be transposed.  Equivalent to trans='C'
        /// </summary>
        ConjTrans = 113
    }

    public class ComputeGraphMKL : ComputeGraph
    {
        const string mklDllName = "mkl_rt.dll";
        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_sgemm(Order order, Transpose transa, Transpose transb, int m, int n, int k, float alpha, float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc);


        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_scopy(int n, float[] x, int incX, float[] y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vsAdd(int n, float[] a, float[] b, float[] y);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vsMul(int n, float[] a, float[] b, float[] y);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_saxpy(int n, float alpha, float[] x, int incX, float[] y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_sscal(int n, float alpha, float[] x, int incX);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern float cblas_sdot(int n, float[] x, int incX, float[] y, int incY);


        public ComputeGraphMKL(bool needBack = true) 
            : base(needBack)
        {
        }

        public override WeightMatrix mul(WeightMatrix m1, WeightMatrix m2)
        {
            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);

            cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.NoTrans, m1.Rows, m2.Columns, m1.Columns, 1.0f, m1.Weight, m1.Columns, m2.Weight, m2.Columns, 0.0f, res.Weight, m2.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.Trans, m1.Rows, m1.Columns, res.Columns, 1.0f, res.Gradient, res.Columns, m2.Weight, res.Columns, 1.0f, m1.Gradient, m1.Columns);
                    cblas_sgemm(Order.Row, Transpose.Trans, Transpose.NoTrans, m2.Rows, m2.Columns, res.Rows, 1.0f, m1.Weight, m2.Rows, res.Gradient, m2.Columns, 1.0f, m2.Gradient, m2.Columns);
                };
                this.backprop.Add(backward);
            }
            return res;
        }


        public override WeightMatrix add(WeightMatrix m1, WeightMatrix m2)
        {
            var res = weightMatrixFactory.CreateWeightMatrix(m1.Rows, m1.Columns);
            vsAdd(res.Weight.Length, m1.Weight, m2.Weight, res.Weight);


            if (this.needs_backprop)
            {

                Action backward = () =>
                {

                    vsAdd(res.Gradient.Length, res.Gradient, m1.Gradient, m1.Gradient);
                    vsAdd(res.Gradient.Length, res.Gradient, m2.Gradient, m2.Gradient);
                  
                };
                this.backprop.Add(backward);
            }
            return res;

        }



        public override WeightMatrix muladd(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3)
        {
            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);

            cblas_scopy(m3.Weight.Length, m3.Weight, 1, res.Weight, 1);
            cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.NoTrans, m1.Rows, m2.Columns, m1.Columns, 1.0f, m1.Weight, m1.Columns, m2.Weight, m2.Columns, 1.0f, res.Weight, m2.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {

                    vsAdd(m3.Gradient.Length, m3.Gradient, res.Gradient, m3.Gradient);

                    cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.Trans, m1.Rows, m1.Columns, res.Columns, 1.0f, res.Gradient, res.Columns, m2.Weight, res.Columns, 1.0f, m1.Gradient, m1.Columns);
                    cblas_sgemm(Order.Row, Transpose.Trans, Transpose.NoTrans, m2.Rows, m2.Columns, res.Rows, 1.0f, m1.Weight, m2.Rows, res.Gradient, m2.Columns, 1.0f, m2.Gradient, m2.Columns);
             
                };
                this.backprop.Add(backward);
            }
            return res;
        }







        //public override WeightMatrix eltmul(WeightMatrix m1, WeightMatrix m2)
        //{

        //    var res = new WeightMatrix(m1.Rows, m1.Columns);
        //    var n = m1.Weight.Length;

        //    vsMul(n, m1.Weight, m2.Weight, res.Weight);


        //    if (this.needs_backprop)
        //    {
        //        Action backward = () =>
        //        {
        //            float[] tmp = new float[m1.Gradient.Length];
        //            vsMul(n, m2.Weight, res.Gradient, tmp);
        //            vsAdd(n, m1.Gradient, tmp, m1.Gradient);

        //            vsMul(n, m1.Weight, res.Gradient, tmp);
        //            vsAdd(n, m2.Gradient, tmp, m2.Gradient);

        //        };
        //        this.backprop.Add(backward);
        //    }
        //    return res;
        //}

        //public override WeightMatrix scalemul(WeightMatrix m1, WeightMatrix m2)
        //{

        //    var res = new WeightMatrix(m1.Rows, m1.Columns);
        //    var n = m1.Weight.Length;

        //    cblas_saxpy(n, m2.Weight[0], m1.Weight, 1, res.Weight, 1);

        //    if (this.needs_backprop)
        //    {

        //        Action backward = () =>
        //        {

        //            cblas_saxpy(n, m2.Weight[0], res.Gradient, 1, m1.Gradient, 1);
        //            m2.Gradient[0] = cblas_sdot(n, m1.Weight, 1, res.Gradient, 1);

                   

        //        };
        //        this.backprop.Add(backward);
        //    }
        //    return res;
        //}
    }
}
