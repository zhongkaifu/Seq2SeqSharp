using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{

    public class Optimizer
    {
        public static float decay_rate = 0.999f;
        public static float smooth_eps = 1e-8f;
        public static float lr_decay_rate = 0.999f;

        public Vector<float> vecDecayRate = new Vector<float>(decay_rate);
        public Vector<float> vecSmoothEPS = new Vector<float>(smooth_eps);

        public float UpdateWeights(List<IWeightMatrix> model, int batchSize, float step_size, float regc, float clipval)
        {
            var vecMaxClipval = new Vector<float>(clipval);
            var vecMinClipval = new Vector<float>(-clipval);

            float AvgAllLearningRate = 0.0f;
            foreach (var m in model)
            {
                m.AvgLearningRate = 0.0f;

                if (m is WeightTensor)
                {
                    if (m.RowToBeUpdated.Count == 0)
                    {
                        UpdateWeightsTensor(m as WeightTensor, batchSize, step_size, clipval, regc);
                    }
                    else
                    {
                        foreach (var kv in m.RowToBeUpdated)
                        {
                            int rowId = kv.Key;
                            int bs = kv.Value;
                            UpdateWeightsTensor(m as WeightTensor, bs, step_size, clipval, regc, rowId);
                        }

                        m.RowToBeUpdated.Clear();
                    }
                }
                else
                {
                    UpdateWeightsCPU(step_size, regc, clipval, vecMaxClipval, vecMinClipval, m as WeightMatrix);
                }

                AvgAllLearningRate += m.AvgLearningRate;

            }

            AvgAllLearningRate /= model.Count;

            return step_size;
        }

        private void UpdateWeightsCPU(float step_size, float regc, float clipval, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, WeightMatrix m)
        {
            if (m.RowToBeUpdated.Count == 0)
            {
                UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Weight.Length, 0);

                m.AvgLearningRate /= m.Weight.Length;
            }
            else
            {
                foreach (var kv in m.RowToBeUpdated)
                {
                    int rowId = kv.Key;
                    UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Columns, rowId * m.Columns);
                }

                m.AvgLearningRate /= (m.RowToBeUpdated.Count * m.Columns);

                m.RowToBeUpdated.Clear();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc)
        {
            Ops.Mul(m.TGradient, m.TGradient, 1.0f / batchSize);
            Ops.Clamp(m.TGradient, m.TGradient, -clipval, clipval);
            Ops.UpdateCash(m.TCash, m.TCash, m.TGradient, decay_rate);

            Ops.UpdateDelta(m.TGradient, m.TGradient, m.TCash, smooth_eps);

            Ops.UpdateCash(m.TLrW, m.TLrW, m.TGradient, lr_decay_rate);

            Ops.UpdateWeight2(m.TWeight, m.TWeight, m.TGradient, m.TLrW, -step_size, -regc);


            Ops.Fill(m.TGradient, 0.0f);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc, int rowId)
        {
            Tensor TWeight = m.TWeight.Narrow(0, rowId, 1);
            Tensor TGradient = m.TGradient.Narrow(0, rowId, 1);
            Tensor TCash = m.TCash.Narrow(0, rowId, 1);
            Tensor TLrW = m.TLrW.Narrow(0, rowId, 1);

            if (batchSize != 1)
            {
                Ops.Mul(TGradient, TGradient, 1.0f / batchSize);
            }

            Ops.Clamp(TGradient, TGradient, -clipval, clipval);
            Ops.UpdateCash(TCash, TCash, TGradient, decay_rate);

            Ops.UpdateDelta(TGradient, TGradient, TCash, smooth_eps);

 
            Ops.UpdateCash(TLrW, TLrW, TGradient, lr_decay_rate);

            Ops.UpdateWeight2(TWeight, TWeight, TGradient, TLrW, -step_size, -regc);

            Ops.Fill(TGradient, 0.0f);


            TWeight.Dispose();
            TGradient.Dispose();
            TCash.Dispose();
            TLrW.Dispose();
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeights(float step_size, float regc, float clipval, WeightMatrix m, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, int n, int i)
        {
            var s = m.Cash;
            var l = m.LrW;
            var vecBaseLR = new Vector<float>(step_size);

            var moreItems = (n % Vector<float>.Count);
            while (i < n - moreItems)
            {
                var vecMDWI = new Vector<float>(m.Gradient, i);

                vecMDWI = Vector.Min(vecMDWI, vecMaxClipval);
                vecMDWI = Vector.Max(vecMDWI, vecMinClipval);

                var vecS = new Vector<float>(s, i);
                vecS = vecS * vecDecayRate + (Vector<float>.One - vecDecayRate) * vecMDWI * vecMDWI;
                vecS.CopyTo(s, i);

                var vecMDWIDelta = vecMDWI / Vector.SquareRoot(vecS + vecSmoothEPS);
                var vecLRWeight = new Vector<float>(l, i);
                var vecLR = ComputeLearningRate(vecMDWIDelta, ref vecLRWeight, vecBaseLR);
                vecLRWeight.CopyTo(l, i);

                var vecMW = new Vector<float>(m.Weight, i);
                var vecDelta = -vecLR * vecMDWIDelta - regc * vecMW;

                vecMW += vecDelta;
                vecMW.CopyTo(m.Weight, i);

                Vector<float>.Zero.CopyTo(m.Gradient, i);


                m.AvgLearningRate += Vector.Dot(vecLR, Vector<float>.One);


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

                var wDelta = (float)(mdwi / Math.Sqrt(s[i] + smooth_eps));
                var lr = ComputeLearningRate(wDelta, l, i, step_size);

                var delta = (float)(-lr * wDelta - regc * m.Weight[i]);

                // update (and regularize)
                m.Weight[i] += delta;

                m.Gradient[i] = 0; // reset gradients for next iteration


                m.AvgLearningRate += lr;


                i++;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeLearningRate(float delta, float[] m, int i, float baseLR)
        {
            var dg = m[i] + delta * delta;
            m[i] = dg;

            return (float)(baseLR / (1.0 + Math.Sqrt(dg)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> ComputeLearningRate(Vector<float> vecDelta, ref Vector<float> vecWeightLearningRate, Vector<float> vecBaseLR)
        {
            var dg = vecWeightLearningRate + vecDelta * vecDelta;
            vecWeightLearningRate = dg;

            return vecBaseLR / (Vector.SquareRoot(dg) + Vector<float>.One);

        }

        public void CleanCash(List<IWeightMatrix> model)
        {
            foreach (var k in model)
            {
                k.CleanCash();
            }
        }
    }
}
