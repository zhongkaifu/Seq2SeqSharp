using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class SparseWeightMatrix
    {
        public int Rows { get; set; }
        public int Columns { get; set; }

        public Dictionary<int, Dictionary<int, float>> Weights = new Dictionary<int, Dictionary<int, float>>();
        public Dictionary<int, Dictionary<int, float>> Gradient = new Dictionary<int, Dictionary<int, float>>();

        public SparseWeightMatrix(int r, int c)
        {
            Rows = r;
            Columns = c;
            
        }

        public void AddWeight(int r, int c, float v)
        {
            if (Weights.ContainsKey(r) == false)
            {
                Weights.Add(r, new Dictionary<int, float>());
            }

            if (Gradient.ContainsKey(r) == false)
            {
                Gradient.Add(r, new Dictionary<int, float>());
            }

            Weights[r][c] = v;
            Gradient[r][c] = 0.0f;
        }
    }
}
