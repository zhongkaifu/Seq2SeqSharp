// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Enums;
using System;
using System.ComponentModel.DataAnnotations;
using System.Globalization;

namespace Seq2SeqSharp.Applications
{
    public class ImageCaptionOptions : Seq2SeqOptions
    {
        [Arg("Input image height in pixels.", nameof(ImageHeight))]
        [Range(1, 16384)]
        public int ImageHeight = 224;

        [Arg("Input image width in pixels.", nameof(ImageWidth))]
        [Range(1, 16384)]
        public int ImageWidth = 224;

        [Arg("Number of image channels (1 for grayscale, 3 for RGB, 4 for RGBA).", nameof(ImageChannels))]
        [Range(1, 4)]
        public int ImageChannels = 3;

        [Arg("Kernel size used in CNN encoder. Must be an odd value.", nameof(CnnKernelSize))]
        [Range(1, 31)]
        public int CnnKernelSize = 3;

        [Arg("Stride used in CNN encoder.", nameof(CnnStride))]
        [Range(1, 16)]
        public int CnnStride = 2;

        [Arg("Base channel multiplier for CNN encoder.", nameof(CnnChannelBase))]
        [Range(1, 1024)]
        public int CnnChannelBase = 64;

        [Arg("Comma separated per-channel mean for image normalization.", nameof(ImageNormalizeMean))]
        public string ImageNormalizeMean = "0.485,0.456,0.406";

        [Arg("Comma separated per-channel standard deviation for image normalization.", nameof(ImageNormalizeStd))]
        public string ImageNormalizeStd = "0.229,0.224,0.225";

        internal float[] ParsedImageNormalizeMean { get; private set; }
        internal float[] ParsedImageNormalizeStd { get; private set; }

        public ImageCaptionOptions()
        {
            EncoderType = EncoderTypeEnums.VisionCNN;
            DecoderType = DecoderTypeEnums.Transformer;
            SharedEmbeddings = false;
        }

        public override void ValidateOptions()
        {
            base.ValidateOptions();

            if (EncoderType != EncoderTypeEnums.VisionCNN)
            {
                throw new ArgumentException("Image caption task requires the VisionCNN encoder.");
            }

            if (DecoderType != DecoderTypeEnums.Transformer)
            {
                throw new ArgumentException("Image caption task currently only supports the Transformer decoder.");
            }

            if (PointerGenerator)
            {
                throw new ArgumentException("Pointer generator is not supported for image caption tasks.");
            }

            if (AMP)
            {
                throw new ArgumentException("AMP is temporarily disabled for vision workloads.");
            }

            if (CnnKernelSize % 2 == 0)
            {
                throw new ArgumentException("CNN kernel size must be an odd value.");
            }

            if (ImageChannels != 1 && ImageChannels != 3 && ImageChannels != 4)
            {
                throw new ArgumentException("Image channels must be 1, 3 or 4.");
            }

            ParsedImageNormalizeMean = ParseNormalizationValues(ImageNormalizeMean, ImageChannels, nameof(ImageNormalizeMean));
            ParsedImageNormalizeStd = ParseNormalizationValues(ImageNormalizeStd, ImageChannels, nameof(ImageNormalizeStd));
        }

        private static float[] ParseNormalizationValues(string rawValues, int expectedLength, string argumentName)
        {
            if (string.IsNullOrWhiteSpace(rawValues))
            {
                throw new ArgumentException($"{argumentName} cannot be empty.");
            }

            var splits = rawValues.Split(',', StringSplitOptions.RemoveEmptyEntries);
            if (splits.Length != expectedLength)
            {
                throw new ArgumentException($"{argumentName} must contain exactly {expectedLength} comma separated values.");
            }

            var result = new float[expectedLength];
            for (int i = 0; i < expectedLength; i++)
            {
                if (!float.TryParse(splits[i], NumberStyles.Float, CultureInfo.InvariantCulture, out result[i]))
                {
                    throw new ArgumentException($"Failed to parse value '{splits[i]}' in {argumentName}.");
                }

                if (argumentName == nameof(ImageNormalizeStd) && result[i] == 0.0f)
                {
                    throw new ArgumentException("Image normalization standard deviation must be non-zero.");
                }
            }

            return result;
        }
    }
}
