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
        List<SelfAttention> m_encoders = new List<SelfAttention>();
        private int m_batchSize;
        private int m_inputDim;

        public TransformerEncoder(string name, int batchSize, int multiHeadNum, int hiddenDim, int inputDim, int depth, int deviceId)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}', batchSize = '{batchSize}'");

            m_batchSize = batchSize;
            m_inputDim = inputDim;
            for (int i = 0; i < depth; i++)
            {
                m_encoders.Add(new SelfAttention($"{name}.SelfAttn_{i}", batchSize, multiHeadNum, hiddenDim, inputDim, deviceId));
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
        public IWeightTensor Encode(IWeightTensor rawInput, IComputeGraph g)
        {        
            int seqLen = rawInput.Rows / m_batchSize;
            var posEmbedding = g.BuildPositionMatrix(seqLen, m_inputDim);
            var posEmbeddingRepeat = g.RepeatRows(posEmbedding, m_batchSize);

            // Transpose to batch-first based sequence
            var inputs = g.TransposeBatch(rawInput, m_batchSize);

            inputs = g.Mul(inputs, (float)Math.Sqrt(m_inputDim));
            inputs = g.Add(inputs, posEmbeddingRepeat);

            for (int k = 0; k < m_encoders.Count; k++)
            {
                inputs = m_encoders[k].Perform(inputs, g);
            }

            // Transpose back to time-first based sequence
            rawInput = g.TransposeBatch(inputs, seqLen);

            return rawInput;
        }



        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (var item in m_encoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in m_encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in m_encoders)
            {
                item.Load(stream);
            }
        }
    }
}
