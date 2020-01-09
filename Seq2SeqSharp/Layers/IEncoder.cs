using Seq2SeqSharp.Tools;

namespace Seq2SeqSharp
{
    public interface IEncoder : INeuralUnit
    {
        IWeightTensor Encode(IWeightTensor rawInput, int batchSize, IComputeGraph g);
        void Reset(IWeightFactory weightFactory, int batchSize);
    }
}
