using System.IO;

namespace Seq2SeqSharp
{
    public interface IMultiProcessorNetworkWrapper
    {
        void Save(IModelMetaData model);
        void Load(IModelMetaData model);
        void SyncWeights();
        void SumGradientsToNetworkOnDefaultDevice();
        INeuralUnit GetNeuralUnitOnDefaultDevice();
        void ZeroGradientsOnAllDevices();
        void ReleaseGradientsOnAllDevices();
    }
}
