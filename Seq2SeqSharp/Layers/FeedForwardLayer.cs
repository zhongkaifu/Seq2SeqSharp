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
    class FeedForwardLayer : INeuralUnit
    {
        private IWeightTensor m_Whd;
        private IWeightTensor m_Bd;
        private string m_name;
        private float m_dropoutRatio;
        private int m_inputDim;
        private int m_outputDim;
        private int m_deviceId;

        public FeedForwardLayer(string name, int inputDim, int outputDim, float dropoutRatio, int deviceId)
        {
            Logger.WriteLine($"Create feed forward layer '{name}' InputDim = '{inputDim}', OutputDim = '{outputDim}', DropoutRatio = '{dropoutRatio}', DeviceId = '{deviceId}'");

            m_name = name;
            m_inputDim = inputDim;
            m_outputDim = outputDim;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;

            m_Whd = new WeightTensor(new long[2] { inputDim, outputDim }, deviceId, name: $"{name}.{nameof(m_Whd)}", isTrainable: true, normal: true);
            m_Bd = new WeightTensor(new long[2] { 1, outputDim }, 0, deviceId, name: $"{name}.{nameof(m_Bd)}", isTrainable: true);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public IWeightTensor Process(IWeightTensor inputT, int batchSize, IComputeGraph graph)
        {
            var g = graph.CreateSubGraph(m_name);
            var res = g.Affine(inputT, m_Whd, m_Bd);
            return g.Dropout(res, batchSize, m_dropoutRatio, inPlace: true);
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();
            response.Add(m_Whd);
            response.Add(m_Bd);

            return response;
        }

        public void Save(Stream stream)
        {
            m_Whd.Save(stream);
            m_Bd.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_Whd.Load(stream);
            m_Bd.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new FeedForwardLayer(m_name, m_inputDim, m_outputDim, m_dropoutRatio, deviceId);
        }
    }
}
