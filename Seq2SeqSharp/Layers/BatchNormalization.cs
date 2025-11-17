// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    /// <summary>
    /// Lightweight batch normalization for 1-D activations (BatchSeq x Channels).
    /// Used by the CNN encoder to stabilize every convolutional layer.
    /// </summary>
    internal sealed class BatchNormalization : INormalization
    {
        private readonly IWeightTensor m_gamma;
        private readonly IWeightTensor m_beta;
        private readonly IWeightTensor m_runningMean;
        private readonly IWeightTensor m_runningVar;
        private readonly float m_epsilon;
        private readonly float m_momentum;
        private readonly string m_name;
        private readonly int m_dim;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly float m_learningRateFactor;
        private readonly DType m_elementType;

        public BatchNormalization(string name, int dim, int deviceId, bool isTrainable, float learningRateFactor = 1.0f, float epsilon = 1e-5f, float momentum = 0.1f, DType elementType = DType.Float32)
        {
            if (dim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(dim));
            }

            m_name = name;
            m_dim = dim;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;
            m_learningRateFactor = learningRateFactor;
            m_elementType = elementType;
            m_epsilon = epsilon;
            m_momentum = momentum;

            m_gamma = new WeightTensor(new long[2] { 1, dim }, 1.0f, deviceId, name: $"{name}.{nameof(m_gamma)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_beta = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_beta)}", isTrainable: isTrainable, learningRateFactor: learningRateFactor, dtype: elementType);
            m_runningMean = new WeightTensor(new long[2] { 1, dim }, 0, deviceId, name: $"{name}.{nameof(m_runningMean)}", isTrainable: false, needGradient: false, dtype: elementType);
            m_runningVar = new WeightTensor(new long[2] { 1, dim }, 1, deviceId, name: $"{name}.{nameof(m_runningVar)}", isTrainable: false, needGradient: false, dtype: elementType);
        }

        public IWeightTensor Norm(IWeightTensor input, IComputeGraph g)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input));
            }

            bool reshapeRequired = input.Sizes.Length != 2;
            IWeightTensor working = input;
            if (reshapeRequired)
            {
                long channelDim = input.Sizes[^1];
                if (channelDim <= 0)
                {
                    throw new ArgumentException("The last dimension must be positive for batch normalization.");
                }

                long batchSeq = input.ElementCount / channelDim;
                working = g.View(input, dims: new long[] { batchSeq, channelDim });
            }

            long[] workingShape = working.Sizes;
            bool useBatchStatistics = m_isTrainable && g.NeedsBackprop;

            IWeightTensor mean;
            IWeightTensor variance;
            IWeightTensor centered;

            if (useBatchStatistics)
            {
                mean = g.Mean(working, 0);
                var expandedMean = g.Expand(mean, workingShape);
                centered = g.Sub(working, expandedMean);
                expandedMean.ReleaseWeight();

                var squared = g.Pow(centered, 2.0f);
                variance = g.Mean(squared, 0);
                squared.ReleaseWeight();

                UpdateRunningStatistic(m_runningMean, mean);
                UpdateRunningStatistic(m_runningVar, variance);
            }
            else
            {
                mean = m_runningMean.CopyWeightsRef($"{m_name}.RunningMean.Infer", needGradient: false, graphToBind: g);
                var expandedMean = g.Expand(mean, workingShape);
                centered = g.Sub(working, expandedMean);
                expandedMean.ReleaseWeight();

                variance = m_runningVar.CopyWeightsRef($"{m_name}.RunningVar.Infer", needGradient: false, graphToBind: g);
            }

            var varianceWithEps = g.Add(variance, m_epsilon);
            var invStd = g.Rsqrt(varianceWithEps);
            varianceWithEps.ReleaseWeight();

            var invStdExpanded = g.Expand(invStd, workingShape);
            var normalized = g.EltMul(centered, invStdExpanded);
            centered.ReleaseWeight();
            invStdExpanded.ReleaseWeight();
            invStd.ReleaseWeight();

            var gammaExpanded = g.Expand(m_gamma, workingShape);
            var scaled = g.EltMul(normalized, gammaExpanded);
            gammaExpanded.ReleaseWeight();
            normalized.ReleaseWeight();

            var betaExpanded = g.Expand(m_beta, workingShape);
            var shifted = g.Add(scaled, betaExpanded);
            betaExpanded.ReleaseWeight();
            scaled.ReleaseWeight();

            mean.ReleaseWeight();
            variance.ReleaseWeight();

            if (reshapeRequired)
            {
                var restored = g.View(shifted, dims: input.Sizes);
                shifted.ReleaseWeight();
                working.ReleaseWeight();
                return restored;
            }

            return shifted;
        }

        private void UpdateRunningStatistic(IWeightTensor runningTensor, IWeightTensor batchStatistic)
        {
            if (!m_isTrainable)
            {
                return;
            }

            float[] running = runningTensor.ToWeightArray();
            float[] batch = batchStatistic.ToWeightArray();
            for (int i = 0; i < running.Length; i++)
            {
                running[i] = (1 - m_momentum) * running[i] + m_momentum * batch[i];
            }

            runningTensor.SetWeightArray(running);
        }

        public List<IWeightTensor> GetParams()
        {
            return new List<IWeightTensor>
            {
                m_gamma,
                m_beta,
                m_runningMean,
                m_runningVar
            };
        }

        public void Save(IModel model)
        {
            m_gamma.Save(model);
            m_beta.Save(model);
            m_runningMean.Save(model);
            m_runningVar.Save(model);
        }

        public void Load(IModel model)
        {
            m_gamma.Load(model);
            m_beta.Load(model);
            m_runningMean.Load(model);
            m_runningVar.Load(model);
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new BatchNormalization(m_name, m_dim, deviceId, m_isTrainable, m_learningRateFactor, m_epsilon, m_momentum, m_elementType);
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }
    }
}
