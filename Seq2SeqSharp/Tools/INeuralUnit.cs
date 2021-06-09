using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp
{
    public interface INeuralUnit
    {
        List<IWeightTensor> GetParams();
        void Save(IModelMetaData model);
        void Load(IModelMetaData model);

        INeuralUnit CloneToDeviceAt(int deviceId);
        int GetDeviceId();
    }
}
