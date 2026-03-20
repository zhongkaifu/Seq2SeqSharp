using System;
using System.Collections.Generic;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TensorSharp;
using Seq2SeqSharp.Tools;

namespace Seq2SeqSharp.Utils
{
    public static class ImageTensorBuilder
    {
        public static float[] LoadImages(IReadOnlyList<string> imagePaths, int channels, int height, int width, float[] normalizeMean = null, float[] normalizeStd = null)
        {
            var normalizedMean = PrepareNormalization(normalizeMean, channels, nameof(normalizeMean));
            var normalizedStd = PrepareNormalization(normalizeStd, channels, nameof(normalizeStd));
            float[] buffer = new float[imagePaths.Count * channels * height * width];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                string path = imagePaths[i];
                if (string.IsNullOrEmpty(path))
                {
                    throw new ArgumentException("Image path cannot be null or empty.");
                }

                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"Image '{path}' was not found.");
                }

                using Image<Rgba32> image = Image.Load<Rgba32>(path);
                image.Mutate(ctx => ctx.Resize(width, height));

                int baseOffset = i * channels * height * width;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Rgba32 pixel = image[x, y];
                        float r = pixel.R / 255f;
                        float g = pixel.G / 255f;
                        float b = pixel.B / 255f;
                        float a = pixel.A / 255f;
                        int spatialIndex = y * width + x;

                        if (channels == 1)
                        {
                            buffer[baseOffset + spatialIndex] = Normalize((r + g + b) / 3.0f, 0, normalizedMean, normalizedStd);
                        }
                        else if (channels >= 3)
                        {
                            buffer[baseOffset + spatialIndex] = Normalize(r, 0, normalizedMean, normalizedStd);
                            buffer[baseOffset + height * width + spatialIndex] = Normalize(g, 1, normalizedMean, normalizedStd);
                            buffer[baseOffset + 2 * height * width + spatialIndex] = Normalize(b, 2, normalizedMean, normalizedStd);
                            if (channels > 3)
                            {
                                buffer[baseOffset + 3 * height * width + spatialIndex] = Normalize(a, 3, normalizedMean, normalizedStd);
                            }
                        }
                        else
                        {
                            throw new NotSupportedException($"Unsupported channel count '{channels}'.");
                        }
                    }
                }
            }

            return buffer;
        }

        public static IWeightTensor BuildTensor(IComputeGraph graph, IReadOnlyList<string> imagePaths, int channels, int height, int width, float[] normalizeMean, float[] normalizeStd, string name)
        {
            if (graph == null)
            {
                throw new ArgumentNullException(nameof(graph));
            }

            var factory = graph.GetWeightFactory();
            var tensor = factory.CreateWeightTensor(new long[] { imagePaths.Count, channels, height, width }, graph.DeviceId, DType.Float32, cleanWeights: true, name: name, needGradient: false);
            var data = LoadImages(imagePaths, channels, height, width, normalizeMean, normalizeStd);
            tensor.SetWeightArray(data);
            return tensor;
        }

        private static float[] PrepareNormalization(float[] values, int channels, string paramName)
        {
            if (values == null || values.Length == 0)
            {
                return null;
            }

            if (values.Length != channels)
            {
                throw new ArgumentException($"Normalization parameter '{paramName}' expected {channels} values but received {values.Length}.");
            }

            return values;
        }

        private static float Normalize(float value, int channelIdx, float[] mean, float[] std)
        {
            if (mean != null)
            {
                value -= mean[channelIdx];
            }

            if (std != null)
            {
                float denom = std[channelIdx];
                if (denom == 0.0f)
                {
                    throw new ArgumentException("Image normalization standard deviation must be non-zero.");
                }

                value /= denom;
            }

            return value;
        }
    }
}
