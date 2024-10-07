using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Layers
{
    internal interface IAttentionLayer
    {
        IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor keyMask, int batchSize, IComputeGraph graph, Dictionary<string, IWeightTensor> cachedTensors = null);
        (IWeightTensor, IWeightTensor) Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph, bool outputAttenWeights = false, Dictionary<string, IWeightTensor> cachedTensors = null);
        List<IWeightTensor> GetParams();
        void Save(IModel stream);
        void Load(IModel stream);
    }
}
