using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightMatrix : IDisposable
    {
        int Rows { get; set; }
        int Columns { get; set; }
        float AvgLearningRate { get; set; }

        Dictionary<int, int> RowToBeUpdated { get; set; }

        void CleanCash();

        float GetWeightAt(int offset);
        void SetWeightAt(float val, int offset);
        void SetGradientAt(float val, int offset);

        void SetGradientByWeight(IWeightMatrix src);

        void SetWeightAtRow(int row, float[] val);

        float[] ToWeightArray();
        int GetMaxWeightIdx();

        void SetWeightArray(float[] v);

        void ReleaseWeight();

        void ClearGradient();
        void ClearWeight();

        void Save(Stream stream);
        void Load(Stream stream);
    }
}
