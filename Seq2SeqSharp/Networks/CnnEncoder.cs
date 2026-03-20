using System;
using System.Collections.Generic;
using Seq2SeqSharp.Tools;
using TensorSharp;
using TensorSharp.Cpu;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp
{
    internal class CnnEncoder : IEncoder
    {
        private readonly List<WeightTensor> m_convWeights = new List<WeightTensor>();
        private readonly List<WeightTensor> m_convBiases = new List<WeightTensor>();
        private readonly int m_deviceId;
        private readonly int m_kernelSize;
        private readonly int m_stride;
        private readonly int m_padding;
        private readonly int m_layerDepth;
        private readonly int m_hiddenDim;
        private readonly int m_imageHeight;
        private readonly int m_imageWidth;
        private readonly int m_channels;
        private readonly float m_dropoutRatio;
        private readonly string m_name;
        private readonly bool m_isTrainable;
        private readonly int m_channelBase;
        private readonly List<LayerNormalization> m_normLayers = new List<LayerNormalization>();
        private readonly List<int> m_outputHeightsPerLayer = new List<int>();
        private readonly List<int> m_outputWidthsPerLayer = new List<int>();
        private int m_outputHeight;
        private int m_outputWidth;

        public CnnEncoder(string name, int imageChannels, int hiddenDim, int layerDepth, int kernelSize, int stride, int channelBase, int imageHeight, int imageWidth, float dropoutRatio, int deviceId, bool isTrainable)
        {
            if (imageHeight <= 0 || imageWidth <= 0)
            {
                throw new ArgumentException("Image height and width must be positive.");
            }

            if (kernelSize <= 0 || kernelSize % 2 == 0)
            {
                throw new ArgumentException("Kernel size must be positive odd number.");
            }

            if (stride <= 0)
            {
                throw new ArgumentException("Stride must be positive.");
            }

            m_name = name;
            m_deviceId = deviceId;
            m_kernelSize = kernelSize;
            m_stride = stride;
            m_padding = kernelSize / 2;
            m_layerDepth = layerDepth;
            m_hiddenDim = hiddenDim;
            m_channels = imageChannels;
            m_imageHeight = imageHeight;
            m_imageWidth = imageWidth;
            m_dropoutRatio = dropoutRatio;
            m_isTrainable = isTrainable;
            m_channelBase = channelBase;

            InitializeLayers();
        }

        private void InitializeLayers()
        {
            int inChannels = m_channels;
            int height = m_imageHeight;
            int width = m_imageWidth;

            for (int i = 0; i < m_layerDepth; i++)
            {
                int plannedChannels = m_channelBase * (1 << i);
                if (plannedChannels > m_hiddenDim)
                {
                    plannedChannels = m_hiddenDim;
                }

                if (i == m_layerDepth - 1)
                {
                    plannedChannels = m_hiddenDim;
                }

                var weight = new WeightTensor(new long[] { plannedChannels, inChannels, m_kernelSize, m_kernelSize }, m_deviceId, name: $"{m_name}.ConvW_{i}", isTrainable: m_isTrainable, initType: RandomInitType.Normal, fanIn: true, fanOut: true);
                var bias = new WeightTensor(new long[] { plannedChannels }, m_deviceId, name: $"{m_name}.ConvB_{i}", isTrainable: m_isTrainable, initType: RandomInitType.Uniform);
                var norm = new LayerNormalization($"{m_name}.ConvNorm_{i}", plannedChannels, m_deviceId, m_isTrainable);

                m_convWeights.Add(weight);
                m_convBiases.Add(bias);
                m_normLayers.Add(norm);

                height = ComputeConvDim(height);
                width = ComputeConvDim(width);
                m_outputHeightsPerLayer.Add(height);
                m_outputWidthsPerLayer.Add(width);
                inChannels = plannedChannels;
            }

            if (height <= 0 || width <= 0)
            {
                throw new ArgumentException("Invalid CNN configuration: output spatial size is non-positive.");
            }

            m_outputHeight = height;
            m_outputWidth = width;
        }

        private int ComputeConvDim(int size)
        {
            return (size + 2 * m_padding - m_kernelSize) / m_stride + 1;
        }

        public int GetDeviceId()
        {
            return m_deviceId;
        }

        public void Reset(IWeightFactory weightFactory, int batchSize)
        {
        }

        public IWeightTensor Encode(IWeightTensor rawInput, int batchSize, IComputeGraph g, IWeightTensor srcSelfMask)
        {
            if (rawInput == null)
            {
                throw new ArgumentNullException(nameof(rawInput));
            }

            if (rawInput.Sizes.Length != 4)
            {
                throw new ArgumentException("Vision encoder expects input shape [batch, channels, height, width].");
            }

            IWeightTensor current = rawInput;
            for (int i = 0; i < m_convWeights.Count; i++)
            {
                var conv = g.Conv2D(current, m_convWeights[i], m_convBiases[i], strideW: m_stride, strideH: m_stride, padW: m_padding, padH: m_padding);
                current.ReleaseWeight();

                long[] convShape = (long[])conv.Sizes.Clone();
                long flattenedLength = (long)batchSize * m_outputHeightsPerLayer[i] * m_outputWidthsPerLayer[i];
                long channelSize = convShape[1];
                var flattened = g.View(conv, new long[] { flattenedLength, channelSize });
                var normalized = m_normLayers[i].Norm(flattened, g);
                flattened.ReleaseWeight();
                var reshaped = g.View(normalized, convShape);
                conv.ReleaseWeight();
                normalized.ReleaseWeight();
                var activated = g.SiLU(reshaped, inPlace: true);
                if (!object.ReferenceEquals(activated, reshaped))
                {
                    reshaped.ReleaseWeight();
                }

                current = activated;
            }

            if (m_dropoutRatio > 0)
            {
                current = g.Dropout(current, m_dropoutRatio, inPlace: true);
            }

            var permuted = g.Permute(current, 0, 2, 3, 1);
            current.ReleaseWeight();
            var flattened = g.View(permuted, dims: new long[] { batchSize * m_outputHeight * m_outputWidth, m_hiddenDim });
            permuted.ReleaseWeight();
            return flattened;
        }

        public INeuralUnit CloneToDeviceAt(int deviceId)
        {
            return new CnnEncoder(m_name, m_channels, m_hiddenDim, m_layerDepth, m_kernelSize, m_stride, m_channelBase, m_imageHeight, m_imageWidth, m_dropoutRatio, deviceId, m_isTrainable);
        }

        public List<IWeightTensor> GetParams()
        {
            List<IWeightTensor> tensors = new List<IWeightTensor>();
            tensors.AddRange(m_convWeights);
            tensors.AddRange(m_convBiases);
            foreach (var norm in m_normLayers)
            {
                tensors.AddRange(norm.GetParams());
            }
            return tensors;
        }

        public void Save(IModel model)
        {
            foreach (var tensor in m_convWeights)
            {
                tensor.Save(model);
            }

            foreach (var tensor in m_convBiases)
            {
                tensor.Save(model);
            }

            foreach (var norm in m_normLayers)
            {
                norm.Save(model);
            }
        }

        public void Load(IModel model)
        {
            foreach (var tensor in m_convWeights)
            {
                tensor.Load(model);
            }

            foreach (var tensor in m_convBiases)
            {
                tensor.Load(model);
            }

            foreach (var norm in m_normLayers)
            {
                norm.Load(model);
            }
        }
    }
}
