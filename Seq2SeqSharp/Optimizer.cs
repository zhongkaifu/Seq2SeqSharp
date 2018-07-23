using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks; 
namespace Seq2SeqSharp
{

    public class Optimizer
    {
        public static float decay_rate = 0.999f;
        public static float smooth_eps = 1e-8f;

        public Vector<float> vecDecayRate = new Vector<float>(decay_rate);
        public Vector<float> vecSmoothEPS = new Vector<float>(smooth_eps);

        public void UpdateWeights(List<WeightMatrix> model, float step_size, float regc, float clipval)
        {
            var vecMaxClipval = new Vector<float>(clipval);
            var vecMinClipval = new Vector<float>(-clipval);

            Parallel.ForEach(model, m =>
            {
                if (m.RowToBeUpdated.Count == 0)
                {
                    var n = m.Weight.Length;
                    var i = 0;
                    UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, n, i);
                }
                else
                {
                    foreach (int rowId in m.RowToBeUpdated)
                    {
                        var n = m.Columns;
                        var i = rowId * m.Columns;
                        UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, n, i);
                    }

                    m.RowToBeUpdated.Clear();
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeights(float step_size, float regc, float clipval, WeightMatrix m, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, int n, int i)
        {
            var s = m.Cash;
            var moreItems = (n % Vector<float>.Count);
            while (i < n - moreItems)
            {
                var vecMDWI = new Vector<float>(m.Gradient, i);

                vecMDWI = Vector.Min(vecMDWI, vecMaxClipval);
                vecMDWI = Vector.Max(vecMDWI, vecMinClipval);

                var vecS = new Vector<float>(s, i);
                vecS = vecS * vecDecayRate + (Vector<float>.One - vecDecayRate) * vecMDWI * vecMDWI;
                vecS.CopyTo(s, i);

                var vecMW = new Vector<float>(m.Weight, i);
                var vecDelta = -step_size * vecMDWI / Vector.SquareRoot(vecS + vecSmoothEPS) - regc * vecMW;

                vecMW += vecDelta;
                vecMW.CopyTo(m.Weight, i);

                Vector<float>.Zero.CopyTo(m.Gradient, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                // rmsprop adaptive learning rate
                var mdwi = m.Gradient[i];
                // gradient clip
                if (mdwi > clipval)
                {
                    mdwi = clipval;
                }
                if (mdwi < -clipval)
                {
                    mdwi = -clipval;
                }

                s[i] = (float)(s[i] * decay_rate + (1.0 - decay_rate) * mdwi * mdwi);
                var delta = (float)(-step_size * mdwi / Math.Sqrt(s[i] + smooth_eps) - regc * m.Weight[i]);

                // update (and regularize)
                m.Weight[i] += delta;

                m.Gradient[i] = 0; // reset gradients for next iteration
                i++;
            }
        }

        public void CleanCash(List<WeightMatrix> model)
        {
            Parallel.ForEach(model, k =>
            {
                k.Cash = new float[k.Cash.Length];
            });
        }
    }
}
