using AdvUtils;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Layers
{
    internal class MoEFeedForward : IFeedForwardLayer
    {
        private readonly LayerNormalization layerNorm;
        private readonly IWeightTensor m_Whd1;
        private readonly IWeightTensor m_Router;
        private readonly IWeightTensor m_Whd2;

        private readonly string m_name;
        private readonly int m_expertNum;
        private readonly int m_hiddenDim;

        private ActivateFuncEnums m_activateFunc;

        public MoEFeedForward(string name, int expertNum, int hiddenDim, float dropoutRatio, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, ActivateFuncEnums activateFunc = ActivateFuncEnums.Relu)
        {
            m_name = name;
            m_activateFunc = activateFunc;
            m_expertNum = expertNum;
            m_hiddenDim = hiddenDim;

            Logger.WriteLine($"Creating MoE feed forward layer. Name = '{name}', ExpertNum = '{expertNum}', HiddenDim = '{hiddenDim}', DeviceId = '{deviceId}', Dropout ratio = '{dropoutRatio}', IsTrainable = '{isTrainable}', Learning rate factor = '{learningRateFactor}', Activate Function = '{activateFunc}'");

            layerNorm = new LayerNormalization($"{name}.{nameof(layerNorm)}", hiddenDim, deviceId, isTrainable, learningRateFactor: learningRateFactor);

            m_Whd1 = new WeightTensor(new long[3] { expertNum, hiddenDim, hiddenDim * 4 }, deviceId, name: $"{name}.{nameof(m_Whd1)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_Whd2 = new WeightTensor(new long[3] { expertNum, hiddenDim * 4, hiddenDim }, deviceId, name: $"{name}.{nameof(m_Whd2)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);
            m_Router = new WeightTensor(new long[2] { hiddenDim, expertNum}, deviceId, name: $"{name}.{nameof(m_Router)}", normType: NormType.Uniform, isTrainable: isTrainable, learningRateFactor: learningRateFactor);

        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            throw new NotImplementedException();
        }

        public int GetDeviceId()
        {
            throw new NotImplementedException();
        }

        public IWeightTensor Process(IWeightTensor input, int batchSize, IComputeGraph graph)
        {
            int seqLen = input.Rows / batchSize;
            using var g = graph.CreateSubGraph($"{m_name}_MoEFeedForward");
            var inputNorm = layerNorm.Norm(input, g);

            var inputRouter = g.Mul(inputNorm, m_Router); // [batchSize * seqLen, expertNum]
            inputRouter = g.Softmax(inputRouter); // [batchSize * seqLen, expertNum]
            inputRouter = g.View(inputRouter, dims: new long[] { batchSize, seqLen, m_expertNum });
            inputRouter = g.Transpose(inputRouter, 2, 0); // [expertNum, seqLen, batchSize]
            inputRouter = g.Transpose(inputRouter, 1, 2); // [expertNum, batchSize, seqLen]
            inputRouter = g.AsContiguous(inputRouter); // [expertNum, batchSize, seqLen]

            int K = (int)(seqLen / m_expertNum + 1);
            (var topKValue, var topKIndex) = g.TopK(inputRouter, K); // [expertNum, batchSize, K]

            float[] factors = new float[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                factors[i] = i * seqLen;
            }
            var factorTensor = g.CreateTensorWeights(sizes: new long[] { 1, batchSize, 1 }, factors);
            factorTensor = g.Expand(factorTensor, dims: new long[] { m_expertNum, batchSize, K });
            topKIndex = g.Add(topKIndex, factorTensor);

            topKIndex = g.AsContiguous(g.View(topKIndex, dims: new long[] { m_expertNum * batchSize * K, 1 }));
            topKIndex.UnbindFromComputeGraph();

            var selectedEmbs = g.IndexSelect(inputNorm, topKIndex, clearWeights: true); // [expertNum * batchSize * K, hiddenDim]
            selectedEmbs = g.View(selectedEmbs, dims: new long[] { m_expertNum, batchSize * K, -1 }); // [expertNum, batchSize * K, hiddenDim];
            selectedEmbs = g.MulBatch(selectedEmbs, m_Whd1); // [expertNum, batchSize * K, hiddenDim * 4]
            selectedEmbs = ((m_activateFunc == ActivateFuncEnums.Swish) ? g.Swish(selectedEmbs, inPlace: true) : g.Relu(selectedEmbs, inPlace: true));
            selectedEmbs = g.MulBatch(selectedEmbs, m_Whd2); // [expertNum, batchSize * K, hiddenDim]

            topKValue = g.View(topKValue, dims: new long[] { m_expertNum, batchSize * K, 1 });
            topKValue = g.Expand(topKValue, dims: new long[] { m_expertNum, batchSize * K, m_hiddenDim });
            selectedEmbs = g.EltMul(selectedEmbs, topKValue); // [expertNum, batchSize * K, hiddenDim]
            selectedEmbs = g.AsContiguous(g.View(selectedEmbs, dims: new long[] { m_expertNum * batchSize * K, m_hiddenDim }));

            var outputEmbs = g.IndexUpdate(input.Sizes, selectedEmbs, topKIndex, true); // [batchSize * seqLen, hiddenDim]
            outputEmbs = graph.Add(outputEmbs, input);

            return outputEmbs;
        }
      
        public virtual List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();

            response.AddRange(layerNorm.GetParams());
            response.AddRange(m_Whd1.GetParams());
            response.AddRange(m_Whd2.GetParams());
            response.AddRange(m_Router.GetParams());

            return response;
        }


        public void Save(IModel stream)
        {
            layerNorm.Save(stream);
            m_Whd1.Save(stream);
            m_Whd2.Save(stream);
            m_Router.Save(stream);

            stream.AddWeights($"{m_name}.ActivateFunc", new float[1] { (float)m_activateFunc });
        }


        public void Load(IModel stream)
        {
            layerNorm.Load(stream);
            m_Whd1.Load(stream);
            m_Whd2.Load(stream);
            m_Router.Load(stream);

            m_activateFunc = (ActivateFuncEnums)stream.GetWeights($"{m_name}.ActivateFunc")[0];
            Logger.WriteLine($"Loading '{m_name}' activate function setting '{m_activateFunc}'");

        }


    }
}
