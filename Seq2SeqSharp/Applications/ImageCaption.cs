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
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.LearningRate;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Applications
{
    /// <summary>
    /// Thin convenience wrapper that exposes a strongly-typed workflow for the image caption task.
    /// It internally relies on the generic <see cref="Seq2Seq"/> implementation but guarantees that
    /// the underlying model is configured with <see cref="ImageCaptionOptions"/>.
    /// </summary>
    public sealed class ImageCaption
    {
        private readonly Seq2Seq m_seq2Seq;

        public ImageCaption(ImageCaptionOptions options)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            m_seq2Seq = new Seq2Seq(options);
        }

        public ImageCaption(ImageCaptionOptions options, Vocab tgtVocab)
        {
            if (options == null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            m_seq2Seq = new Seq2Seq(options, null, tgtVocab);
        }

        public event EventHandler StatusUpdateWatcher
        {
            add => m_seq2Seq.StatusUpdateWatcher += value;
            remove => m_seq2Seq.StatusUpdateWatcher -= value;
        }

        public event EventHandler EvaluationWatcher
        {
            add => m_seq2Seq.EvaluationWatcher += value;
            remove => m_seq2Seq.EvaluationWatcher -= value;
        }

        public event EventHandler EpochEndWatcher
        {
            add => m_seq2Seq.EpochEndWatcher += value;
            remove => m_seq2Seq.EpochEndWatcher -= value;
        }

        public void Train(int maxTrainingEpoch, ICorpus<IPairBatch> trainCorpus, ICorpus<IPairBatch>[] validCorpusList,
            ILearningRate learningRate, IMetric[] metrics, IOptimizer optimizer, DecodingOptions decodingOptions)
        {
            m_seq2Seq.Train(maxTrainingEpoch, trainCorpus, validCorpusList, learningRate, metrics, optimizer, decodingOptions);
        }

        public void ValidVision(ICorpus<IVisionSntPairBatch> validCorpus, List<IMetric> metrics, DecodingOptions decodingOptions)
        {
            m_seq2Seq.ValidVision(validCorpus, metrics, decodingOptions);
        }

        public void TestVision(string inputImageListFile, string outputFile, int batchSize, DecodingOptions decodingOptions,
            string tgtSpmPath, string outputAlignmentFile = null)
        {
            m_seq2Seq.TestVision(inputImageListFile, outputFile, batchSize, decodingOptions, tgtSpmPath, outputAlignmentFile);
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_seq2Seq.DumpVocabToFiles(outputSrcVocab, outputTgtVocab);
        }

        public void UpdateVocabs(Vocab srcVocab, Vocab tgtVocab)
        {
            m_seq2Seq.UpdateVocabs(srcVocab, tgtVocab);
        }
    }
}
