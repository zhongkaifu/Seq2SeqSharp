
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
    public class AttentionDecoder
    {
        public List<LSTMAttentionDecoderCell> decoders = new List<LSTMAttentionDecoderCell>(); 
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }
        public AttentionUnit attentionLayer { get; set; }

        public AttentionDecoder(int sdim, int hdim, int dim, int depth)
        {
            attentionLayer = new AttentionUnit(hdim);
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;

            decoders.Add(new LSTMAttentionDecoderCell(sdim, hdim, dim));
            for (int i = 1; i < depth; i++)
            {
                decoders.Add(new LSTMAttentionDecoderCell(0, hdim, hdim));
            }
        }
        public void Reset()
        {
            foreach (var item in decoders)
            {
                item.Reset();
            }

        }

        public void PreProcess(IWeightMatrix encoderOutput, IComputeGraph g)
        {
            attentionLayer.PreProcess(encoderOutput, g);
        }


        public IWeightMatrix Decode(SparseWeightMatrix sparseInput, IWeightMatrix input, IWeightMatrix encoderOutput, IComputeGraph g)
        {
            var V = input;
            var lastStatus = this.decoders.FirstOrDefault().ct;
            var context = attentionLayer.Perform(encoderOutput, lastStatus, g);

            foreach (var decoder in decoders)
            {
                var e = decoder.Step(sparseInput, context, V, g);
                V = e;
            }

            return V;
        } 

        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in decoders)
            {
                response.AddRange(item.getParams());
            }
            response.AddRange(attentionLayer.getParams());

            return response;
        }

        //public List<float[]> GetWeightList()
        //{
        //    List<float[]> wl = new List<float[]>();

        //    foreach (var item in decoders)
        //    {
        //        wl.AddRange(item.GetWeightList());
        //    }

        //    wl.AddRange(attentionLayer.GetWeightList());

        //    return wl;
        //}

        //public void SetWeightList(List<float[]> wl)
        //{
        //    foreach (var item in decoders)
        //    {
        //        item.SetWeightList(wl);
        //    }

        //    attentionLayer.SetWeightList(wl);
        //}

    }
}
