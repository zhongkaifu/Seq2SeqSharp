using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{

    [Serializable]
    public class AttentionUnit
    {

        public WeightMatrix V { get; set; }
        public WeightMatrix Ua { get; set; }
        public WeightMatrix bUa { get; set; }
        public WeightMatrix Wa { get; set; }
        public WeightMatrix bWa { get; set; }
        public int MaxIndex { get; set; }
        public AttentionUnit()
        {
        }
        public AttentionUnit(int size)
        {
            this.Ua = new WeightMatrix((size * 2)  , size, true);

            this.Wa = new WeightMatrix(size  , size, true);

             
            this.bUa = new WeightMatrix(1, size, 0);
            this.bWa = new WeightMatrix(1, size, 0);

            this.V = new WeightMatrix(size, 1, true);
        }


        private object locker = new object();
        public WeightMatrix Perform(List<WeightMatrix> input, WeightMatrix state, IComputeGraph g)
        {
            WeightMatrix context;
            WeightMatrix[] atten = new WeightMatrix[input.Count];

            var wc = g.muladd(state, Wa, bWa);
            Parallel.For(0, input.Count, i =>
            {
                var h_j = input[i];
                var uh = g.muladd(h_j, Ua, bUa);
                var gg = g.addtanh(uh, wc);
                var aa = g.mul(gg, V);

                atten[i] = aa;
            });

            var res = g.Softmax(atten);
            var cmax = res[0].Weight[0];
            int maxAtt = 0;
            for (int i = 1; i < res.Count; i++)
            {
                if (res[i].Weight[0] > cmax)
                {
                    cmax = res[i].Weight[0];
                    maxAtt = i;
                }
            }
            this.MaxIndex = maxAtt;


            context = g.scalemul(input[0], res[0]);
            for (int hj = 1; hj < input.Count; hj++)
            {
                context = g.scalemuladd(input[hj], res[hj], context);
            }
            return context;
        }

        public WeightMatrix Perform(WeightMatrix input, WeightMatrix state, IComputeGraph g)
        {
            WeightMatrix context;
            List<WeightMatrix> atten = new List<WeightMatrix>();

            var stateRepeat = g.RepeatRows(state, input.Rows);
            var baiseInput = new WeightMatrix(input.Rows, 1, 1);
            var inputb = g.concatColumns(input, baiseInput);


            var uh = g.mul(inputb, Ua);


            baiseInput = new WeightMatrix(stateRepeat.Rows, 1, 1);
            stateRepeat = g.concatColumns(stateRepeat, baiseInput);


            var wc = g.mul(stateRepeat, Wa);
            var gg = g.addtanh(uh, wc);
            var aa = g.mul(gg, V);


            var res = g.Softmax(aa);

              
            var weighted = g.weightRows(input, res); ;
            context = g.sumColumns(weighted);

            return context; 
        }

        public virtual List<WeightMatrix> getParams()
        {
            List<WeightMatrix> response = new List<WeightMatrix>();
            response.Add(this.Ua);
            response.Add(this.Wa);
            response.Add(this.bUa);
            response.Add(this.bWa);
            response.Add(this.V);
            return response;
        }
    }
}
