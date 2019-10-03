using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightTensor : IDisposable
    {
        int Rows { get; set; }
        int Columns { get; set; }
        string Name { get; set; }

        bool IsTrainable { get; set; }

        int DeviceId { get; set; }

        void CleanCache();

        float GetWeightAt(int offset);
        void SetWeightAt(float val, int offset);

        void SetGradientByWeight(IWeightTensor src);

        void SetWeightAtRow(int row, float[] val);

        float[] ToWeightArray();

        List<int> GetTopNMaxWeightIdx(int topN);

        void SetWeightArray(float[] v);

        void ReleaseWeight();
        void ReleaseGradient();

        void ClearGradient();
        void ClearWeight();

        void Save(Stream stream);
        void Load(Stream stream);

        void CopyWeights(IWeightTensor src);
        void AddGradient(IWeightTensor src);
    }
}
