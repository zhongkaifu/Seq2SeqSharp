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

        public Vector<float> vecDecayRate = new Vector<float>(decay_rate);
        public Vector<float> vecSmoothEPS = new Vector<float>(smooth_eps);

        public float UpdateWeights(List<IWeightMatrix> model, float step_size, float regc, float clipval, bool updatedLR = true)
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
                        UpdateWeightsTensor(m as WeightTensor, step_size, clipval, regc, updatedLR);
                    }
                    else
                    {
                        foreach (int rowId in m.RowToBeUpdated)
                        {
                            UpdateWeightsTensor(m as WeightTensor, step_size, clipval, regc, updatedLR, rowId);
                        }

                        m.RowToBeUpdated.Clear();
                    }
                }
                else
                {
                    UpdateWeightsCPU(step_size, regc, clipval, updatedLR, vecMaxClipval, vecMinClipval, m as WeightMatrix);
                }

                AvgAllLearningRate += m.AvgLearningRate;

            }

            AvgAllLearningRate /= model.Count;

            return AvgAllLearningRate;
        }

        private void UpdateWeightsCPU(float step_size, float regc, float clipval, bool updatedLR, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, WeightMatrix m)
        {
            if (m.RowToBeUpdated.Count == 0)
            {
                UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Weight.Length, 0, updatedLR);

                m.AvgLearningRate /= m.Weight.Length;
            }
            else
            {
                foreach (int rowId in m.RowToBeUpdated)
                {
                    UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Columns, rowId * m.Columns, updatedLR);
                }

                m.AvgLearningRate /= (m.RowToBeUpdated.Count * m.Columns);

                m.RowToBeUpdated.Clear();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, float step_size, float clipval, float regc, bool updateLR)
        {
            Ops.Clamp(m.TGradient, m.TGradient, -clipval, clipval);
            Ops.UpdateCash(m.TCash, m.TCash, m.TGradient, decay_rate);

            Tensor tDelta = new Tensor(TensorAllocator.Allocator, DType.Float32, m.Rows, m.Columns);
            Ops.UpdateDelta(tDelta, m.TGradient, m.TCash, smooth_eps);

            Tensor tLR = null;
            if (updateLR)
            {
                Ops.AddMul(m.TLrW, m.TLrW, tDelta, tDelta);
                tLR = Ops.RsqrtOne(null, m.TLrW, -step_size);
            }
            else
            {
                tLR = Ops.UpdateLR(null, tDelta, m.TLrW, -step_size);
            }
            m.AvgLearningRate += -Ops.SumAll(tLR);

            Ops.UpdateWeight(m.TWeight, m.TWeight, tDelta, tLR, -regc);

            Ops.Fill(m.TGradient, 0.0f);

            m.AvgLearningRate /= (m.Rows * m.Columns);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, float step_size, float clipval, float regc, bool updateLR, int rowId)
        {
            Tensor TWeight = m.TWeight.Narrow(0, rowId, 1);
            Tensor TGradient = m.TGradient.Narrow(0, rowId, 1);
            Tensor TCash = m.TCash.Narrow(0, rowId, 1);
            Tensor TLrW = m.TLrW.Narrow(0, rowId, 1);

            Ops.Clamp(TGradient, TGradient, -clipval, clipval);           
            Ops.UpdateCash(TCash, TCash, TGradient, decay_rate);

            Tensor tDelta = new Tensor(TensorAllocator.Allocator, DType.Float32, 1, m.Columns);
            Ops.UpdateDelta(tDelta, TGradient, TCash, smooth_eps);

            Tensor tLR = null;
            if (updateLR)
            {
                Ops.AddMul(TLrW, TLrW, tDelta, tDelta);
                tLR = Ops.RsqrtOne(null, TLrW, -step_size);
            }
            else
            {
                tLR = Ops.UpdateLR(null, tDelta, TLrW, -step_size);
            }
            m.AvgLearningRate += -Ops.SumAll(tLR);

            Ops.UpdateWeight(TWeight, TWeight, tDelta, tLR, -regc);

            Ops.Fill(TGradient, 0.0f);

            m.AvgLearningRate /= (1 * m.Columns);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeights(float step_size, float regc, float clipval, WeightMatrix m, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, int n, int i, bool updateLR = true)
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
                var vecLR = ComputeLearningRate(vecMDWIDelta, ref vecLRWeight, vecBaseLR, updateLR);
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
                var lr = ComputeLearningRate(wDelta, l, i, step_size, updateLR);

                var delta = (float)(-lr * wDelta - regc * m.Weight[i]);

                // update (and regularize)
                m.Weight[i] += delta;

                m.Gradient[i] = 0; // reset gradients for next iteration


                m.AvgLearningRate += lr;


                i++;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeLearningRate(float delta, float[] m, int i, float baseLR, bool updateLR = true)
        {
            var dg = m[i] + delta * delta;

            if (updateLR)
            {
                m[i] = dg;
            }

            return (float)(baseLR / (1.0 + Math.Sqrt(dg)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> ComputeLearningRate(Vector<float> vecDelta, ref Vector<float> vecWeightLearningRate, Vector<float> vecBaseLR, bool updateLR = true)
        {
            var dg = vecWeightLearningRate + vecDelta * vecDelta;

            if (updateLR)
            {
                vecWeightLearningRate = dg;
            }

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
