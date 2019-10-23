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
        private int m_inputDim;
        private float m_dropoutRatio;

        public TransformerEncoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'");

            m_inputDim = inputDim;
            m_dropoutRatio = dropoutRatio;

            for (int i = 0; i < depth; i++)
            {
                m_encoders.Add(new SelfAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId));
            }
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
        }

        /// <summary>
        /// Transformer encoder
        /// </summary>
        /// <param name="rawInputs"></param>
        /// <param name="g"></param>
        /// <returns></returns>
        public IWeightTensor Encode(IWeightTensor rawInput, int batchSize, IComputeGraph g)
        {        
            int seqLen = rawInput.Rows / batchSize;
            var posEmbedding = g.BuildPositionMatrix(seqLen, m_inputDim);
            var posEmbeddingRepeat = g.RepeatRows(posEmbedding, batchSize, runGradient: false);

            // Transpose to batch-first based sequence
            var inputs = g.TransposeBatch(rawInput, batchSize);

            //inputs = g.Mul(inputs, (float)Math.Sqrt(m_inputDim));
            //inputs = g.Add(inputs, posEmbeddingRepeat, runGradient1: true, runGradient2: false);

            inputs = g.AddMul(posEmbeddingRepeat, inputs, (float)Math.Sqrt(m_inputDim), runGradientW1: false, runGradientW2: true);

            // We don't update position embedding, so dispose it now to save memory.
            posEmbeddingRepeat.Dispose();
            posEmbedding.Dispose();

            inputs = g.Dropout(inputs, batchSize, m_dropoutRatio, inPlace: true);

            for (int k = 0; k < m_encoders.Count; k++)
            {
                inputs = m_encoders[k].Perform(inputs, batchSize, g);
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
