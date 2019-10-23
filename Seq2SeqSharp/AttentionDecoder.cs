
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
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
        private string m_name;

        public AttentionDecoder(string name, int hdim, int dim, int context, int depth, int deviceId)
        {
            attentionLayer = new AttentionUnit($"{name}.AttnUnit", hdim, context, deviceId);
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
            m_name = name;

            decoders.Add(new LSTMAttentionDecoderCell($"{name}.LSTMAttn_0", hdim, dim, context, deviceId));
            for (int i = 1; i < depth; i++)
            {
                decoders.Add(new LSTMAttentionDecoderCell($"{name}.LSTMAttn_{i}", hdim, hdim, context, deviceId));
            }
        }


        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in decoders)
            {
                item.Reset(weightFactory, batchSize);
            }

        }

        public AttentionPreProcessResult PreProcess(IWeightTensor encoderOutput, int batchSize, IComputeGraph g)
        {
            return attentionLayer.PreProcess(encoderOutput, batchSize, g);
        }


        public IWeightTensor Decode(IWeightTensor input, AttentionPreProcessResult attenPreProcessResult, int batchSize, IComputeGraph g)
        {
            var V = input;
            var lastStatus = this.decoders.LastOrDefault().Cell;
            var context = attentionLayer.Perform(lastStatus, attenPreProcessResult, batchSize, g);

            foreach (var decoder in decoders)
            {
                var e = decoder.Step(context, V, g);
                V = e;
            }

            return V;
        }

        public List<IWeightTensor> GetCTs()
        {
            List<IWeightTensor> res = new List<IWeightTensor>();
            foreach (var decoder in decoders)
            {
                res.Add(decoder.Cell);
            }

            return res;
        }

        public List<IWeightTensor> GetHTs()
        {
            List<IWeightTensor> res = new List<IWeightTensor>();
            foreach (var decoder in decoders)
            {
                res.Add(decoder.Hidden);
            }

            return res;
        }

        public void SetCTs(List<IWeightTensor> l)
        {
            for (int i = 0; i < l.Count; i++)
            {
                decoders[i].Cell = l[i];
            }
        }

        public void SetHTs(List<IWeightTensor> l)
        {
            for (int i = 0; i < l.Count; i++)
            {
                decoders[i].Hidden = l[i];
            }
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (var item in decoders)
            {
                response.AddRange(item.getParams());
            }
            response.AddRange(attentionLayer.getParams());

            return response;
        }

        public void Save(Stream stream)
        {
            attentionLayer.Save(stream);
            foreach (var item in decoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            attentionLayer.Load(stream);
            foreach (var item in decoders)
            {
                item.Load(stream);
            }
        }
    }
}
