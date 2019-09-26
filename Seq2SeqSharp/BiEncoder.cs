
using AdvUtils;
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
    public class BiEncoder : IEncoder
    {
        private List<LSTMCell> m_forwardEncoders;
        private List<LSTMCell> m_backwardEncoders;

        private int m_hiddenDim;
        private int m_inputDim;
        private int m_depth;

        private int m_batchSize;

        public BiEncoder(int batchSize, int hiddenDim, int inputDim, int depth, int deviceId)
        {
            Logger.WriteLine($"Creating BiLSTM encoder at device '{deviceId}'. HiddenDim = '{hiddenDim}', InputDim = '{inputDim}', Depth = '{depth}'");

            m_forwardEncoders = new List<LSTMCell>();
            m_backwardEncoders = new List<LSTMCell>();

            m_forwardEncoders.Add(new LSTMCell(batchSize, hiddenDim, inputDim, deviceId));
            m_backwardEncoders.Add(new LSTMCell(batchSize, hiddenDim, inputDim, deviceId));

            for (int i = 1; i < depth; i++)
            {
                m_forwardEncoders.Add(new LSTMCell(batchSize, hiddenDim, hiddenDim * 2, deviceId));
                m_backwardEncoders.Add(new LSTMCell(batchSize, hiddenDim, hiddenDim * 2, deviceId));
            }

            m_hiddenDim = hiddenDim;
            m_inputDim = inputDim;
            m_depth = depth;
            m_batchSize = batchSize;
        }

        public void Reset(IWeightFactory weightFactory)
        {
            foreach (var item in m_forwardEncoders)
            {
                item.Reset(weightFactory);
            }

            foreach (var item in m_backwardEncoders)
            {
                item.Reset(weightFactory);
            }
        }

        public IWeightTensor Encode(IWeightTensor rawInputs, IComputeGraph g)
        {
            int seqLen = rawInputs.Rows / m_batchSize;

            List<IWeightTensor> inputs = new List<IWeightTensor>();
            for (int i = 0; i < seqLen; i++)
            {
                var emb_i = g.PeekRow(rawInputs, i * m_batchSize, m_batchSize);
                inputs.Add(emb_i);
            }

            List<IWeightTensor> forwardOutputs = new List<IWeightTensor>();
            List<IWeightTensor> backwardOutputs = new List<IWeightTensor>();

            List<IWeightTensor> layerOutputs = inputs.ToList();
            for (int i = 0; i < m_depth; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    var forwardOutput = m_forwardEncoders[i].Step(layerOutputs[j], g);
                    forwardOutputs.Add(forwardOutput);

                    var backwardOutput = m_backwardEncoders[i].Step(layerOutputs[inputs.Count - j - 1], g);
                    backwardOutputs.Add(backwardOutput);
                }

                backwardOutputs.Reverse();
                layerOutputs.Clear();
                for (int j = 0; j < seqLen; j++)
                {
                    var concatW = g.ConcatColumns(forwardOutputs[j], backwardOutputs[j]);
                    layerOutputs.Add(concatW);
                }

            }

            return g.ConcatRows(layerOutputs);
        }


        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            foreach (var item in m_forwardEncoders)
            {
                response.AddRange(item.getParams());
            }


            foreach (var item in m_backwardEncoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in m_forwardEncoders)
            {
                item.Save(stream);
            }

            foreach (var item in m_backwardEncoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in m_forwardEncoders)
            {
                item.Load(stream);
            }

            foreach (var item in m_backwardEncoders)
            {
                item.Load(stream);
            }
        }
    }
}
