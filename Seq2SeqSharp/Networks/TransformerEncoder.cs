using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class TransformerEncoder : IEncoder
    {
        private readonly List<SelfAttention> m_encoders = new List<SelfAttention>();
        private readonly int m_inputDim;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly int m_multiHeadNum;
        private readonly int m_hiddenDim;
        private readonly int m_depth;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly LayerNormalization layerNorm;

        public TransformerEncoder(string name, int multiHeadNum, int hiddenDim, int inputDim, int depth, float dropoutRatio, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Creating transformer encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}', MultiHeadNum = '{multiHeadNum}'");

            m_name = name;
            m_multiHeadNum = multiHeadNum;
            m_hiddenDim = hiddenDim;
            m_inputDim = inputDim;
            m_depth = depth;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;

            if (hiddenDim != inputDim)
            {
                throw new ArgumentException($"hiddenDim is not equal to inputDim in TransformerEncoder.");
            }

            m_encoders.Add(new SelfAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId, isTrainable: isTrainable));
            for (int i = 1; i < depth; i++)
            {
                m_encoders.Add(new SelfAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable: isTrainable));
            }

            layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable);

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
        public IWeightTensor Encode(IWeightTensor rawInput, IWeightTensor mask, int batchSize, IComputeGraph g)
        {
            int seqLen = rawInput.Rows / batchSize;
            // Transpose to batch-first based sequence
            IWeightTensor inputs = g.TransposeBatch(rawInput, batchSize);

            using (IWeightTensor posEmbedding = g.BuildPositionMatrix(seqLen, m_inputDim))
            {
                using (IWeightTensor posEmbeddingRepeat = g.RepeatRows(posEmbedding, batchSize, runGradient: false))
                {
                    inputs = g.AddMul(posEmbeddingRepeat, inputs, (float)Math.Sqrt(m_inputDim), runGradientW1: false, runGradientW2: true);
                }
            }
            inputs = g.Dropout(inputs, batchSize, m_dropoutRatio, inPlace: true);

            var maskRep = g.RepeatRows(mask, m_multiHeadNum * seqLen, runGradient: false);

            for (int k = 0; k < m_encoders.Count; k++)
            {
                var inputsMultiHeadAtt = m_encoders[k].MultiHeadAttention(inputs, inputs, inputs, maskRep, batchSize, g);
                inputs = m_encoders[k].PositionwiseFeedForward(inputsMultiHeadAtt, batchSize, g);
            }

            inputs = layerNorm.Norm(inputs, g);

            // Transpose back to time-first based sequence
            rawInput = g.TransposeBatch(inputs, seqLen);

            return rawInput;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerEncoder(m_name, m_multiHeadNum, m_hiddenDim, m_inputDim, m_depth, m_dropoutRatio, deviceId, m_isTrainable);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (SelfAttention item in m_encoders)
            {
                response.AddRange(item.getParams());
            }

            response.AddRange(layerNorm.getParams());

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (SelfAttention item in m_encoders)
            {
                item.Save(stream);
            }

            layerNorm.Save(stream);
        }

        public void Load(Stream stream)
        {
            foreach (SelfAttention item in m_encoders)
            {
                item.Load(stream);
            }

            layerNorm.Load(stream);
        }
    }
}
