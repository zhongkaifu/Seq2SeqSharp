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
using System.Globalization;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Seq2SeqSharp.Applications
{
    internal sealed class VisionImageProcessor
    {
        private readonly int m_imageSize;
        private readonly int m_patchSize;
        private readonly int m_gridSize;
        private readonly float[] m_mean;
        private readonly float[] m_std;

        private const int ChannelCount = 3;

        public int PatchCount => m_gridSize * m_gridSize;
        public int FeatureSize => ChannelCount;

        public VisionImageProcessor(Seq2SeqOptions options)
        {
            if (options.VisionImageSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(options.VisionImageSize));
            }

            if (options.VisionPatchSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(options.VisionPatchSize));
            }

            if (options.VisionImageSize % options.VisionPatchSize != 0)
            {
                throw new ArgumentException("VisionImageSize must be divisible by VisionPatchSize so that patches tile the image.");
            }

            m_imageSize = options.VisionImageSize;
            m_patchSize = options.VisionPatchSize;
            m_gridSize = options.VisionImageSize / options.VisionPatchSize;

            m_mean = ParseVector(options.VisionChannelMean, new[] { 0.485f, 0.456f, 0.406f }, nameof(options.VisionChannelMean));
            m_std = ParseVector(options.VisionChannelStd, new[] { 0.229f, 0.224f, 0.225f }, nameof(options.VisionChannelStd));
        }

        public float[] BuildPatchFeatures(IReadOnlyList<string> imagePaths)
        {
            if (imagePaths == null)
            {
                throw new ArgumentNullException(nameof(imagePaths));
            }

            var buffer = new float[imagePaths.Count * PatchCount * FeatureSize];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                EncodeSingleImage(imagePaths[i], buffer, i);
            }

            return buffer;
        }

        private void EncodeSingleImage(string imagePath, float[] buffer, int batchIndex)
        {
            if (string.IsNullOrEmpty(imagePath))
            {
                throw new ArgumentException("Source image path is empty", nameof(imagePath));
            }

            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Cannot locate image '{imagePath}'", imagePath);
            }

            using var image = Image.Load<Rgba32>(imagePath);
            image.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new Size(m_imageSize, m_imageSize),
                Sampler = KnownResamplers.Bicubic
            }));

            int pixelsPerPatch = m_patchSize * m_patchSize;
            float inv = 1.0f / pixelsPerPatch;

            for (int gridY = 0; gridY < m_gridSize; gridY++)
            {
                for (int gridX = 0; gridX < m_gridSize; gridX++)
                {
                    float r = 0;
                    float g = 0;
                    float b = 0;

                    for (int y = 0; y < m_patchSize; y++)
                    {
                        for (int x = 0; x < m_patchSize; x++)
                        {
                            var pixel = image[gridX * m_patchSize + x, gridY * m_patchSize + y];
                            r += pixel.R / 255f;
                            g += pixel.G / 255f;
                            b += pixel.B / 255f;
                        }
                    }

                    int patchIndex = gridY * m_gridSize + gridX;
                    int baseIndex = (batchIndex * PatchCount + patchIndex) * FeatureSize;
                    buffer[baseIndex + 0] = Normalize(r * inv, 0);
                    buffer[baseIndex + 1] = Normalize(g * inv, 1);
                    buffer[baseIndex + 2] = Normalize(b * inv, 2);
                }
            }
        }

        private float Normalize(float value, int channel)
        {
            return (value - m_mean[channel]) / m_std[channel];
        }

        private static float[] ParseVector(string raw, float[] fallback, string argumentName)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                return fallback.ToArray();
            }

            var tokens = raw.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length != ChannelCount)
            {
                throw new ArgumentException($"{argumentName} must contain {ChannelCount} comma separated values.");
            }

            var values = new float[ChannelCount];
            for (int i = 0; i < tokens.Length; i++)
            {
                if (!float.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]))
                {
                    throw new ArgumentException($"Cannot parse value '{tokens[i]}' in {argumentName}.");
                }
            }

            return values;
        }
    }
}
