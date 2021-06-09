using AdvUtils;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    internal class FeedForwardLayer : IFeedForwardLayer
    {
        private readonly IWeightTensor m_Whd;
        private readonly IWeightTensor m_Bd;
        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly int m_inputDim;
        private readonly int m_outputDim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;

        public FeedForwardLayer(string name, int inputDim, int outputDim, float dropoutRatio, int deviceId, bool isTrainable)
        {
            Logger.WriteLine($"Create feed forward layer '{name}' InputDim = '{inputDim}', OutputDim = '{outputDim}', DropoutRatio = '{dropoutRatio}', DeviceId = '{deviceId}'");

            m_name = name;
            m_inputDim = inputDim;
            m_outputDim = outputDim;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;

            m_Whd = new WeightTensor(new long[2] { inputDim, outputDim }, deviceId, name: $"{name}.{nameof(m_Whd)}", normType: NormType.Uniform, isTrainable: isTrainable);
            m_Bd = new WeightTensor(new long[2] { 1, outputDim }, 0, deviceId, name: $"{name}.{nameof(m_Bd)}", isTrainable: isTrainable);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public IWeightTensor Process(IWeightTensor inputT, int batchSize, IComputeGraph g)
        {            
            IWeightTensor res = g.Affine(inputT, m_Whd, m_Bd);
            return g.Dropout(res, batchSize, m_dropoutRatio, inPlace: true);
        }

        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>
            {
                m_Whd,
                m_Bd
            };

            return response;
        }

        public void Save(IModelMetaData stream)
        {
            m_Whd.Save(stream);
            m_Bd.Save(stream);
        }


        public void Load(IModelMetaData stream)
        {
            m_Whd.Load(stream);
            m_Bd.Load(stream);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new FeedForwardLayer(m_name, m_inputDim, m_outputDim, m_dropoutRatio, deviceId, m_isTrainable);
        }
    }
}
