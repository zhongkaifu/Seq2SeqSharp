using Seq2SeqSharp.Tools;

namespace Seq2SeqSharp.Layers
{
    public interface INormalization : INeuralUnit
    {
        IWeightTensor Norm(IWeightTensor input, IComputeGraph g);
    }
}
