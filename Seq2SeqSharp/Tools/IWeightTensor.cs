using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightTensor : INeuralUnit, IDisposable
    {
        int Rows { get; set; }
        int Columns { get; set; }
        string Name { get; set; }

        bool IsTrainable { get; set; }

        int DeviceId { get; set; }

        void ZeroCache();

        float GetWeightAt(int offset);
        void SetWeightAt(float val, int offset);

        void CopyWeightsToGradients(IWeightTensor src);

        void SetWeightAtRow(int row, float[] val);

        List<int> GetTopNMaxWeightIdx(int topN);

        void SetWeightArray(float[] v);

        void ReleaseWeight();
        void ReleaseGradient();

        void ZeroGradient();
        void CleanWeight();

        void CopyWeightsFrom(IWeightTensor src);
        void AddGradientFrom(IWeightTensor src);
    }
}
