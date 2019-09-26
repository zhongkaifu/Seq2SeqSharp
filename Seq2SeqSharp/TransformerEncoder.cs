using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    class TransformerEncoder : IEncoder
    {
        List<SelfAttention> encoders = new List<SelfAttention>();
        private int m_batchSize;
        private int m_inputDim;

        public TransformerEncoder(int batchSize, int multiHeadNum, int hiddenDim, int inputDim, int depth, ArchTypeEnums archType, int deviceId)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}', batchSize = '{batchSize}'");

            m_batchSize = batchSize;
            m_inputDim = inputDim;
            for (int i = 0; i < depth; i++)
            {
                encoders.Add(new SelfAttention(batchSize, multiHeadNum, hiddenDim, inputDim, archType, deviceId));
            }
        }

        public void Reset(IWeightFactory weightFactory)
        {
        }

        /// <summary>
        /// Transformer encoder
        /// </summary>
        /// <param name="rawInputs"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightMatrix Encode(IWeightMatrix rawInput, IComputeGraph g)
        {
            int seqLen = rawInput.Rows / m_batchSize;
            var posEmbedding = g.BuildPositionMatrix(seqLen, m_inputDim);
            var posEmbeddingRepeat = g.RepeatRows(posEmbedding, m_batchSize);

            // Transpose to batch-first based sequence
            var inputs = g.PermuteBatch(rawInput, m_batchSize);

            inputs = g.Mul(inputs, (float)Math.Sqrt(m_inputDim));
            inputs = g.Add(inputs, posEmbeddingRepeat);

            for (int k = 0; k < encoders.Count; k++)
            {
                inputs = encoders[k].Perform(inputs, g);
            }

            // Transpose back to time-first based sequence
            rawInput = g.PermuteBatch(inputs, seqLen);

            return rawInput;
        }



        public List<IWeightMatrix> GetParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in encoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Load(stream);
            }
        }
    }
}
