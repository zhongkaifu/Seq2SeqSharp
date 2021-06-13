using System.IO;

namespace Seq2SeqSharp
{
    public interface IMultiProcessorNetworkWrapper
    {
        void Save(IModel model);
        void Load(IModel model);
        void SyncWeights();
        void SumGradientsToNetworkOnDefaultDevice();
        INeuralUnit GetNeuralUnitOnDefaultDevice();
        void ZeroGradientsOnAllDevices();
        void ReleaseGradientsOnAllDevices();
    }
}
