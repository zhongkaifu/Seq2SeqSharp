// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;


namespace Seq2SeqSharp.Corpus
{
    /// <summary>
    /// Data format:
    /// Source side: tokens split by space
    /// Target side: [classification tag] \t tokens split by space
    /// </summary>
    public class Seq2SeqClassificationCorpus : ParallelCorpus<Seq2SeqClassificationCorpusBatch>
    {

        public Seq2SeqClassificationCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
            : base(corpusFilePath, srcLangName, tgtLangName, maxTokenSizePerBatch, maxSrcSentLength, maxTgtSentLength, shuffleEnums: shuffleEnums, tooLongSequence: tooLongSequence)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// For return vocabs: (source vocab, target vocab, classification vocab)
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab, Vocab) BuildVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000, bool sharedVocab = false)
        {
            if (sharedVocab && (srcVocabSize != tgtVocabSize))
            {
                throw new ArgumentException($"Vocab size must be equal if sharedVocab is true. Src Vocab Size = '{srcVocabSize}', Tgt Vocab Size = '{tgtVocabSize}'");
            }

            (CorpusBatch.s_ds, CorpusBatch.t_ds) = CountTokenFreqs();

            CorpusBatch.ReduceSrcTokensToSingleGroup();
            if (sharedVocab)
            {
                CorpusBatch.MergeTokensCountSrcTgt(0, 1);
            }

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(srcVocabSize, tgtVocabSize);

            Vocab srcVocab = srcVocabs[0];
            Vocab clsVocab = tgtVocabs[0];
            Vocab tgtVocab = tgtVocabs[1];

            return (srcVocab, tgtVocab, clsVocab);
        }
    }
}
