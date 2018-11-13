using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;

namespace Seq2SeqSharp
{

    [Serializable]
    public class AttentionUnit
    {

//#if CUDA
//      //  public IWeightMatrix W { get; set; }
//#endif

        public IWeightMatrix V { get; set; }
        public IWeightMatrix Ua { get; set; }
        public IWeightMatrix bUa { get; set; }
        public IWeightMatrix Wa { get; set; }
        public IWeightMatrix bWa { get; set; }

        public AttentionUnit(int size)
        {
#if CUDA
            //ComputeGraphTensor g = new ComputeGraphTensor(false);
            //W = new WeightTensor((size * 2) + size + 1 + 1, size, true);
            //List<IWeightMatrix> Ws = g.SplitRows(W, size * 2, size, 1, 1);
            //Ua = Ws[0];
            //Wa = Ws[1];
            //bUa = Ws[2];
            //bWa = Ws[3];



            this.Ua = new WeightTensor((size * 2), size, true);
            this.Wa = new WeightTensor(size, size, true);
            this.bUa = new WeightTensor(1, size, 0);
            this.bWa = new WeightTensor(1, size, 0);
            this.V = new WeightTensor(size, 1, true);
#else
            this.Ua = new WeightMatrix((size * 2)  , size, true);
            //this.Ua = new WeightMatrix(size, size, true);
            this.Wa = new WeightMatrix(size  , size, true);             
            this.bUa = new WeightMatrix(1, size, 0);
            this.bWa = new WeightMatrix(1, size, 0);

            this.V = new WeightMatrix(size, 1, true);
#endif
        }



        IWeightMatrix bUas;
        IWeightMatrix uhs;
        public void PreProcess(IWeightMatrix inputs, IComputeGraph g)
        {
            bUas = g.RepeatRows(bUa, inputs.Rows);
            uhs = g.MulAdd(inputs, Ua, bUas);

        }

        public IWeightMatrix Perform(IWeightMatrix inputs, IWeightMatrix state, IComputeGraph g)
        {
            var wc = g.MulAdd(state, Wa, bWa);
            var wcs = g.RepeatRows(wc, inputs.Rows);


            //  var inputs = g.ConcatRows(input);
            //var bUas = g.RepeatRows(bUa, inputs.Rows);
            //var uhs = g.MulAdd(inputs, Ua, bUas);



            var ggs = g.AddTanh(uhs, wcs);
            var atten = g.Mul(ggs, V);
            var attenT = g.Transpose2(atten);

            var attenSoftmax = g.Softmax(attenT);

            IWeightMatrix context = g.Mul(attenSoftmax, inputs);
                      
            return context;

        }

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

//#if CUDA
//            response.Add(W);
//#else

            response.Add(this.Ua);
            response.Add(this.Wa);
            response.Add(this.bUa);
            response.Add(this.bWa);
            response.Add(this.V);
//#endif
            return response;
        }

        //public virtual List<float[]> GetWeightList()
        //{
        //    List<float[]> weightList = new List<float[]>();

        //    weightList.Add(Ua.ToWeightArray());
        //    weightList.Add(Wa.ToWeightArray());
        //    weightList.Add(bUa.ToWeightArray());
        //    weightList.Add(bWa.ToWeightArray());
        //    weightList.Add(V.ToWeightArray());

        //    return weightList;

        //}

        //public virtual void SetWeightList(List<float[]> wl)
        //{
        //    Ua.SetWeightArray(wl[0]);
        //    Wa.SetWeightArray(wl[1]);
        //    bUa.SetWeightArray(wl[2]);
        //    bWa.SetWeightArray(wl[3]);
        //    V.SetWeightArray(wl[4]);

        //    wl.RemoveRange(0, 5);

        //}
    }
}
