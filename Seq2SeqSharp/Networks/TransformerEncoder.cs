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
        private string m_name;
        private int m_multiHeadNum;
        private int m_hiddenDim;
        private int m_depth;
        private int m_deviceId;

        public TransformerEncoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'");

            m_name = name;
            m_multiHeadNum = multiHeadNum;
            m_hiddenDim = hiddenDim;
            m_inputDim = inputDim;
            m_depth = depth;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerEncoder.");
            }

            m_encoders.Add(new SelfAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId));
            for (int i = 1; i < depth; i++)
            {
                m_encoders.Add(new SelfAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId));
            }
        }

        public int GetDeviceId()
        {
            return m_deviceId;
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

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerEncoder(m_name, m_multiHeadNum, m_hiddenDim, m_inputDim, m_depth, m_dropoutRatio, deviceId);
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
