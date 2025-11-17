// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using Seq2SeqSharp.Enums;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using TensorSharp;
using System.Linq;

namespace Seq2SeqSharp
{
    internal sealed class CNNEncoder : IEncoder
    {
        private readonly List<IFeedForwardLayer> m_convLayers = new List<IFeedForwardLayer>();
        private readonly List<BatchNormalization> m_batchNormLayers = new List<BatchNormalization>();
        private readonly List<INormalization> m_normLayers = new List<INormalization>();
        private readonly List<IFeedForwardLayer> m_residualProjections = new List<IFeedForwardLayer>();
        private readonly List<int> m_layerInputDims = new List<int>();
        private readonly List<int> m_layerOutputDims = new List<int>();
        private readonly IReadOnlyList<int> m_channelSchedule;
        private readonly int m_hiddenDim;
        private readonly int m_depth;
        private readonly int m_kernelSize;
        private readonly string m_name;
        private readonly float m_dropoutRatio;
        private readonly int m_deviceId;
        private readonly bool m_isTrainable;
        private readonly float m_learningRateFactor;
        private readonly DType m_elementType;
        private readonly NormEnums m_normType;
        private readonly IFeedForwardLayer m_finalProjection;

        public CNNEncoder(string name, int hiddenDim, int depth, int kernelSize, float dropoutRatio, int deviceId, bool isTrainable,
            float learningRateFactor = 1.0f, DType elementType = DType.Float32, NormEnums normType = NormEnums.LayerNorm, IReadOnlyList<int> channelSchedule = null)
        {
            if (kernelSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(kernelSize));
            }

            if (kernelSize % 2 == 0)
            {
                throw new ArgumentException("CNN kernel size must be an odd value so padding stays symmetric.");
            }

            m_name = name;
            m_hiddenDim = hiddenDim;
            m_depth = depth;
            m_kernelSize = kernelSize;
            m_dropoutRatio = dropoutRatio;
            m_deviceId = deviceId;
            m_isTrainable = isTrainable;
            m_learningRateFactor = learningRateFactor;
            m_elementType = elementType;
            m_normType = normType;

            m_channelSchedule = channelSchedule?.ToArray() ?? BuildDefaultChannelSchedule(hiddenDim, depth);
            if (m_channelSchedule.Count != depth)
            {
                throw new ArgumentException($"Channel schedule length ({m_channelSchedule.Count}) must match CNN depth ({depth}).");
            }

            int previousDim = hiddenDim;
            for (int i = 0; i < depth; i++)
            {
                int outputDim = m_channelSchedule[i];
                if (outputDim <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(channelSchedule), "Channel sizes must be positive integers.");
                }

                m_layerInputDims.Add(previousDim);
                m_layerOutputDims.Add(outputDim);

                m_convLayers.Add(new FeedForwardLayer($"{name}.Conv_{i}", previousDim * kernelSize, outputDim, dropoutRatio, deviceId, isTrainable,
                    learningRateFactor: learningRateFactor, elementType: elementType));
                m_batchNormLayers.Add(new BatchNormalization($"{name}.BN_{i}", outputDim, deviceId, isTrainable,
                    learningRateFactor: learningRateFactor, elementType: elementType));

                if (previousDim != outputDim)
                {
                    m_residualProjections.Add(new FeedForwardLayer($"{name}.Proj_{i}", previousDim, outputDim, dropoutRatio, deviceId, isTrainable,
                        learningRateFactor: learningRateFactor, elementType: elementType));
                }
                else
                {
                    m_residualProjections.Add(null);
                }

                if (normType == NormEnums.LayerNorm)
                {
                    m_normLayers.Add(new LayerNormalization($"{name}.Norm_{i}", outputDim, deviceId, isTrainable, learningRateFactor: learningRateFactor,
                        elementType: elementType));
                }
                else
                {
                    m_normLayers.Add(new RMSNormalization($"{name}.Norm_{i}", outputDim, deviceId, isTrainable, learningRateFactor: learningRateFactor,
                        elementType: elementType));
                }

                previousDim = outputDim;
            }

            if (previousDim != hiddenDim)
            {
                m_finalProjection = new FeedForwardLayer($"{name}.FinalProj", previousDim, hiddenDim, dropoutRatio, deviceId, isTrainable,
                    learningRateFactor: learningRateFactor, elementType: elementType);
            }
        }

        public int GetDeviceId() => m_deviceId;

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
        }

        public IWeightTensor Encode(IWeightTensor inputs, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            if (batchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(batchSize));
            }

            if (srcSelfMask != null)
            {
                srcSelfMask.Dispose();
            }

            using IComputeGraph subGraph = g.CreateSubGraph($"{m_name}_CNNEncoder");

            int seqLen = inputs.Rows / batchSize;
            if (seqLen <= 0)
            {
                throw new ArgumentException("Input sequence length must be positive.");
            }

            IWeightTensor states = inputs;
            for (int layerId = 0; layerId < m_depth; layerId++)
            {
                int inDim = m_layerInputDims[layerId];
                int outDim = m_layerOutputDims[layerId];
                var windows = BuildConvWindows(subGraph, states, batchSize, seqLen, inDim);
                var conv = m_convLayers[layerId].Process(windows, batchSize * seqLen, subGraph);
                windows.ReleaseWeight();

                var convWithBatchNorm = m_batchNormLayers[layerId].Norm(conv, subGraph);
                conv.ReleaseWeight();
                conv = subGraph.ReLU(convWithBatchNorm, inPlace: true);
                IWeightTensor residual = states;
                if (m_residualProjections[layerId] != null)
                {
                    residual = m_residualProjections[layerId].Process(states, batchSize * seqLen, subGraph);
                }

                var summed = subGraph.Add(conv, residual);
                conv.ReleaseWeight();
                if (!ReferenceEquals(residual, states))
                {
                    residual.ReleaseWeight();
                }
                states.ReleaseWeight();

                states = m_normLayers[layerId].Norm(summed, subGraph);
                summed.ReleaseWeight();
            }

            if (m_finalProjection != null)
            {
                var projected = m_finalProjection.Process(states, batchSize * seqLen, subGraph);
                states.ReleaseWeight();
                states = projected;
            }

            states.UnbindFromComputeGraph();
            return states;
        }

        private IWeightTensor BuildConvWindows(IComputeGraph g, IWeightTensor source, int batchSize, int seqLen, int channelDim)
        {
            var reshaped = g.View(source, dims: new long[] { batchSize, seqLen, channelDim });
            var flattenedWindows = new List<(IWeightTensor flat, IWeightTensor original)>();
            int radius = m_kernelSize / 2;

            for (int offset = -radius; offset <= radius; offset++)
            {
                IWeightTensor windowSlice = offset == 0 ? reshaped : CreateShiftedSlice(g, reshaped, batchSize, seqLen, offset, channelDim);
                var flat = g.View(windowSlice, dims: new long[] { batchSize * seqLen, channelDim });
                flattenedWindows.Add((flat, windowSlice));
            }

            var concat = g.Concate(flattenedWindows.ConvertAll(item => item.flat), 1);

            foreach (var (flat, original) in flattenedWindows)
            {
                flat.ReleaseWeight();
                if (!ReferenceEquals(original, reshaped))
                {
                    original.ReleaseWeight();
                }
            }

            reshaped.ReleaseWeight();
            return concat;
        }

        private IWeightTensor CreateShiftedSlice(IComputeGraph g, IWeightTensor source, int batchSize, int seqLen, int shift, int channelDim)
        {
            int padLength = Math.Min(Math.Abs(shift), seqLen);
            int remaining = seqLen - padLength;
            var dtype = source.ElementType;
            var pieces = new List<IWeightTensor>();

            if (shift < 0)
            {
                if (padLength > 0)
                {
                    pieces.Add(g.Zero(new long[] { batchSize, padLength, channelDim }, dtype));
                }

                if (remaining > 0)
                {
                    pieces.Add(g.Peek(source, 1, 0, remaining));
                }
            }
            else
            {
                if (remaining > 0)
                {
                    pieces.Add(g.Peek(source, 1, padLength, remaining));
                }

                if (padLength > 0)
                {
                    pieces.Add(g.Zero(new long[] { batchSize, padLength, channelDim }, dtype));
                }
            }

            if (pieces.Count == 0)
            {
                return g.Zero(new long[] { batchSize, seqLen, channelDim }, dtype);
            }

            if (pieces.Count == 1)
            {
                return pieces[0];
            }

            var result = g.Concate(pieces, 1);
            foreach (var piece in pieces)
            {
                piece.ReleaseWeight();
            }

            return result;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new CNNEncoder(m_name, m_hiddenDim, m_depth, m_kernelSize, m_dropoutRatio, deviceId, m_isTrainable,
                learningRateFactor: m_learningRateFactor, elementType: m_elementType, normType: m_normType, channelSchedule: m_channelSchedule);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> response = new List<IWeightTensor>();
            foreach (var conv in m_convLayers)
            {
                response.AddRange(conv.GetParams());
            }

            foreach (var bn in m_batchNormLayers)
            {
                response.AddRange(bn.GetParams());
            }

            foreach (var proj in m_residualProjections)
            {
                if (proj != null)
                {
                    response.AddRange(proj.GetParams());
                }
            }

            foreach (var norm in m_normLayers)
            {
                response.AddRange(norm.GetParams());
            }

            if (m_finalProjection != null)
            {
                response.AddRange(m_finalProjection.GetParams());
            }

            return response;
        }

        public void Save(IModel stream)
        {
            foreach (var conv in m_convLayers)
            {
                conv.Save(stream);
            }

            foreach (var bn in m_batchNormLayers)
            {
                bn.Save(stream);
            }

            foreach (var proj in m_residualProjections)
            {
                proj?.Save(stream);
            }

            foreach (var norm in m_normLayers)
            {
                norm.Save(stream);
            }

            m_finalProjection?.Save(stream);
        }

        public void Load(IModel stream)
        {
            foreach (var conv in m_convLayers)
            {
                conv.Load(stream);
            }

            foreach (var bn in m_batchNormLayers)
            {
                bn.Load(stream);
            }

            foreach (var proj in m_residualProjections)
            {
                proj?.Load(stream);
            }

            foreach (var norm in m_normLayers)
            {
                norm.Load(stream);
            }

            m_finalProjection?.Load(stream);
        }

        private static IReadOnlyList<int> BuildDefaultChannelSchedule(int hiddenDim, int depth)
        {
            if (depth <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(depth));
            }

            if (depth == 1)
            {
                return new int[] { hiddenDim };
            }

            var result = new int[depth];
            float maxGain = hiddenDim >= 768 ? 2.0f : 1.5f;
            for (int layerId = 0; layerId < depth; layerId++)
            {
                float progress = (float)layerId / (depth - 1);
                float sinusoidal = (float)Math.Sin(Math.PI * progress); // 0 at edges, 1 at center
                float gain = 1.0f + sinusoidal * (maxGain - 1.0f);
                int dim = AlignToMultiple((int)Math.Round(hiddenDim * gain), 8);
                if (layerId == depth - 1)
                {
                    dim = hiddenDim;
                }

                result[layerId] = Math.Max(8, dim);
            }

            return result;
        }

        private static int AlignToMultiple(int value, int multiple)
        {
            if (multiple <= 1)
            {
                return value;
            }

            return ((value + multiple - 1) / multiple) * multiple;
        }
    }
}
