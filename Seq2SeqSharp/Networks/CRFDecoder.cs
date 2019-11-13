using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Networks
{
    class CRFDecoder
    {
        public int m_tagSetSize;
        protected float[] CRFWeights { get; set; }
        public const int MINUS_LOG_EPSILON = 13;


        public CRFDecoder(int tagSetSize)
        {
            m_tagSetSize = tagSetSize;
            CRFWeights = new float[m_tagSetSize * m_tagSetSize];
        }

        public void Save(Stream stream)
        {
            // create a byte array and copy the floats into it...
            var byteArray = new byte[CRFWeights.Length * 4];
            Buffer.BlockCopy(CRFWeights, 0, byteArray, 0, byteArray.Length);

            stream.Write(byteArray, 0, byteArray.Length);

        }

        public void Load(Stream stream)
        {
            int size = m_tagSetSize * m_tagSetSize;
            var byteArray = new byte[size * 4];
            stream.Read(byteArray, 0, byteArray.Length);

            CRFWeights = new float[byteArray.Length / 4];
            Buffer.BlockCopy(byteArray, 0, CRFWeights, 0, byteArray.Length);
        }

        public static double logsumexp(double x, double y, bool flg)
        {
            if (flg) return y; // init mode
            var vmin = Math.Min(x, y);
            var vmax = Math.Max(x, y);
            if (vmax > vmin + MINUS_LOG_EPSILON)
            {
                return vmax;
            }

            return vmax + Math.Log(Math.Exp(vmin - vmax) + 1.0);

        }

        public float[] ForwardBackward(int numStates, float[] rawNNOutput)
        {
            double[][] alphaSet = new double[numStates][];
            double[][] betaSet = new double[numStates][];

            //forward
            for (var i = 0; i < numStates; i++)
            {
                alphaSet[i] = new double[m_tagSetSize];
                for (var j = 0; j < m_tagSetSize; j++)
                {
                    double dscore0 = 0;
                    if (i > 0)
                    {
                        for (var k = 0; k < m_tagSetSize; k++)
                        {
                            var fbgm = CRFWeights[j * m_tagSetSize + k];
                            var finit = alphaSet[i - 1][k];
                            var ftmp = fbgm + finit;
                            dscore0 = logsumexp(dscore0, ftmp, k == 0);
                        }
                    }
                    alphaSet[i][j] = dscore0 + rawNNOutput[i * m_tagSetSize + j];
                }
            }

            //backward
            for (var i = numStates - 1; i >= 0; i--)
            {
                betaSet[i] = new double[m_tagSetSize];
                for (var j = 0; j < m_tagSetSize; j++)
                {
                    double dscore0 = 0;
                    if (i < numStates - 1)
                    {
                        for (var k = 0; k < m_tagSetSize; k++)
                        {
                            var fbgm = CRFWeights[k * m_tagSetSize + j];
                            var finit = betaSet[i + 1][k];
                            var ftmp = fbgm + finit;

                            dscore0 = logsumexp(dscore0, ftmp, k == 0);
                        }
                    }
                    betaSet[i][j] = dscore0 + rawNNOutput[i * m_tagSetSize + j];
                }
            }

            //Z_
            double Z_ = 0.0f;
            var betaSet_0 = betaSet[0];
            for (var i = 0; i < m_tagSetSize; i++)
            {
                Z_ = logsumexp(Z_, betaSet_0[i], i == 0);
            }

            //Calculate the output probability of each node

            float[] CRFSeqOutput = new float[numStates * m_tagSetSize];
            for (var i = 0; i < numStates; i++)
            {
                var CRFSeqOutput_i = CRFSeqOutput[i];
                var alphaSet_i = alphaSet[i];
                var betaSet_i = betaSet[i];

                for (var j = 0; j < m_tagSetSize; j++)
                {
                    CRFSeqOutput[i * m_tagSetSize + j] = (float)Math.Exp(alphaSet_i[j] + betaSet_i[j] - rawNNOutput[i * m_tagSetSize + j] - Z_);
                }
            }

            return CRFSeqOutput;
        }


        public class PAIR<T, K>
        {
            public T first;
            public K second;
            public PAIR(T f, K s)            
            {
                first = f;
                second = s;
            }
        }

        public int[][] DecodeNBestCRF(float[] ys, int numStats, int numNBest)
        {
            var vPath = new PAIR<int, int>[numStats, m_tagSetSize, numNBest];
            var DUMP_LABEL = -1;
            var vPreAlpha = new float[m_tagSetSize, numNBest];
            var vAlpha = new float[m_tagSetSize, numNBest];
            var nStartTagIndex = 0;
            var nEndTagIndex = 0;
            var MIN_VALUE = float.MinValue;

            //viterbi algorithm
            for (var i = 0; i < m_tagSetSize; i++)
            {
                for (var j = 0; j < numNBest; j++)
                {
                    vPreAlpha[i, j] = MIN_VALUE;
                    vPath[0, i, j] = new PAIR<int, int>(DUMP_LABEL, 0);
                }
            }

            vPreAlpha[nStartTagIndex, 0] = ys[nStartTagIndex];
            vPath[0, nStartTagIndex, 0].first = nStartTagIndex;

            var q = new PriorityQueue<float, PAIR<int, int>>();
            for (var t = 1; t < numStats; t++)
            {
                for (var j = 0; j < m_tagSetSize; j++)
                {
                    while (q.Count() > 0)
                        q.Dequeue();

                    var _stp = CRFWeights[j * m_tagSetSize];
                    var _y = ys[t * m_tagSetSize + j];
                    for (var k = 0; k < numNBest; k++)
                    {
                        var score = vPreAlpha[0, k] + _stp + _y;
                        q.Enqueue(score, new PAIR<int, int>(0, k));
                    }

                    for (var i = 1; i < m_tagSetSize; i++)
                    {
                        _stp = CRFWeights[j * m_tagSetSize + i];
                        for (var k = 0; k < numNBest; k++)
                        {
                            var score = vPreAlpha[i, k] + _stp + _y;
                            if (score <= q.Peek().Key)
                                break;

                            q.Dequeue();
                            q.Enqueue(score, new PAIR<int, int>(i, k));
                        }
                    }

                    var idx = numNBest - 1;

                    while (q.Count() > 0)
                    {
                        vAlpha[j, idx] = q.Peek().Key;
                        vPath[t, j, idx] = q.Peek().Value;
                        idx--;

                        q.Dequeue();
                    }
                }

                vPreAlpha = vAlpha;
                vAlpha = new float[m_tagSetSize, numNBest];
            }


            //backtrace to get the n-best result path
            var vTagOutput = new int[numNBest][];
            for (var i = 0; i < numNBest; i++)
            {
                vTagOutput[i] = new int[numStats];
            }

            for (var k = 0; k < numNBest; k++)
            {
                vTagOutput[k][numStats - 1] = nEndTagIndex;
                var decision = new PAIR<int, int>(nEndTagIndex, k);

                for (var t = numStats - 2; t >= 0; t--)
                {
                    vTagOutput[k][t] = vPath[t + 1, decision.first, decision.second].first;
                    decision = vPath[t + 1, decision.first, decision.second];
                }
            }

            return vTagOutput;
        }

        public void UpdateBigramTransition(int numStates, float[] CRFSeqOutput, int[] trueTags)
        {
            float[][] crfWeightsDelta = new float[m_tagSetSize][];
            for (int i = 0; i < m_tagSetSize; i++)
            {
                crfWeightsDelta[i] = new float[m_tagSetSize];
            }

            for (var timeat = 1; timeat < numStates; timeat++)
            {
                for (var i = 0; i < m_tagSetSize; i++)
                {
                    var crfWeightsDelta_i = crfWeightsDelta[i];

                    var j = 0;
                    while (j < m_tagSetSize)
                    {
                        crfWeightsDelta_i[j] -= CRFWeights[i * m_tagSetSize + j] * CRFSeqOutput[timeat * m_tagSetSize + i] * CRFSeqOutput[(timeat - 1) * m_tagSetSize + j]; //CRFSeqOutput_pre_timeat[j];
                        j++;
                    }
                }

                var iTagId = trueTags[timeat];
                var iLastTagId = trueTags[timeat - 1];
                crfWeightsDelta[iTagId][iLastTagId] += 1;
            }

            //Update tag Bigram LM
            for (var i = 0; i < m_tagSetSize; i++)
            {
                var CRFWeightsDelta_i = crfWeightsDelta[i];
                var j = 0;
             
                while (j < m_tagSetSize)
                {
                    var delta = CRFWeightsDelta_i[j];
                    var learningRate = 0.001f;
                    //    delta = RNNHelper.NormalizeGradient(delta);
                    CRFWeights[i * m_tagSetSize + j] += learningRate * delta;

                    j++;
                }
            }
        }
    }
}
