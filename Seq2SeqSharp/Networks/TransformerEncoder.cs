using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class TransformerEncoder : IEncoder
    {
        private readonly List<MultiHeadAttention> m_encoders = new List<MultiHeadAttention>();
        private readonly List<PositionwiseFeedForward> m_posFFNs = new List<PositionwiseFeedForward>();

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

            m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_0", multiHeadNum, hiddenDim, inputDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true));
            for (int i = 1; i < depth; i++)
            {
                m_encoders.Add(new MultiHeadAttention($"{name}.SelfAttn_{i}", multiHeadNum, hiddenDim, hiddenDim, m_dropoutRatio, deviceId, isTrainable: isTrainable, sharedQKV: true));              
            }

            for (int i = 0; i < depth; i++)
            {
                m_posFFNs.Add(new PositionwiseFeedForward($"{name}.PosFFN_{i}", hiddenDim, m_dropoutRatio, deviceId, isTrainable));
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
        public IWeightTensor Encode(IWeightTensor inputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            using (IComputeGraph subg = g.CreateSubGraph($"{m_name}_Encoder"))
            {
                IWeightTensor maskTensor = null;
                if (srcSelfMask != null)
                {
                    int seqLen = inputs.Rows / batchSize;
                    using (var keyMaskView = subg.View(srcSelfMask, runGradient: false, dims: new long[] { batchSize, 1, seqLen, seqLen }))
                    {
                        maskTensor = subg.Expand(keyMaskView, runGradient: false, dims: new long[] { batchSize, m_multiHeadNum, seqLen, seqLen });
                    }
                }

                IWeightTensor attnProbs = null;
                for (int k = 0; k < m_encoders.Count; k++)
                {
                    (inputs, attnProbs) = m_encoders[k].Perform(inputs, maskTensor, batchSize, subg, outputAttenWeights: false);
                    inputs = m_posFFNs[k].Perform(inputs, batchSize, subg);
                }

                inputs = layerNorm.Norm(inputs, subg);

                inputs.UnbindFromComputeGraph();
                if (attnProbs != null)
                {
                    attnProbs.UnbindFromComputeGraph();
                }

                if (maskTensor != null)
                {
                    maskTensor.Dispose();
                }
            }

            return inputs;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new TransformerEncoder(m_name, m_multiHeadNum, m_hiddenDim, m_inputDim, m_depth, m_dropoutRatio, deviceId, m_isTrainable);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (MultiHeadAttention item in m_encoders)
            {
                response.AddRange(item.getParams());
            }

            foreach (var item in m_posFFNs)
            {
                response.AddRange(item.getParams());
            }

            response.AddRange(layerNorm.getParams());

            return response;
        }

        public void Save(IModelMetaData stream)
        {
            foreach (MultiHeadAttention item in m_encoders)
            {
                item.Save(stream);
            }

            foreach (var item in m_posFFNs)
            {
                item.Save(stream);
            }

            layerNorm.Save(stream);
        }

        public void Load(IModelMetaData stream)
        {
            foreach (MultiHeadAttention item in m_encoders)
            {
                item.Load(stream);
            }

            foreach (var item in m_posFFNs)
            {
                item.Load(stream);
            }

            layerNorm.Load(stream);
        }
    }
}
