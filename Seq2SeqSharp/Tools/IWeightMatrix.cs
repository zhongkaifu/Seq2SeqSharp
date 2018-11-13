using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public interface IWeightMatrix
    {
        int Rows { get; set; }
        int Columns { get; set; }
        float AvgLearningRate { get; set; }

        HashSet<int> RowToBeUpdated { get; set; }

        void CleanCash();

        float GetWeightAt(int offset);
        void SetWeightAt(float val, int offset);
        void SetGradientAt(float val, int offset);

        void SetGradientByWeight(IWeightMatrix src);

        void SetWeightAtRow(int row, float[] val);

        float[] ToWeightArray();

        void SetWeightArray(float[] v);

        void ClearGradient();
        void ClearWeight();
    }
}
