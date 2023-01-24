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
    public class Seq2SeqCorpus : ParallelCorpus<Seq2SeqCorpusBatch>
    {

        public Seq2SeqCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
            :base (corpusFilePath, srcLangName, tgtLangName, maxTokenSizePerBatch, maxSrcSentLength, maxTgtSentLength, shuffleEnums: shuffleEnums, tooLongSequence: tooLongSequence)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab) BuildVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000, bool sharedVocab = false, int minFreq = 1)
        {
            if (sharedVocab && (srcVocabSize != tgtVocabSize))
            {
                throw new ArgumentException($"Vocab size must be equal if sharedVocab is true. Src Vocab Size = '{srcVocabSize}', Tgt Vocab Size = '{tgtVocabSize}'");
            }

            (CorpusBatch.s_ds, CorpusBatch.t_ds) = CountTokenFreqs();

            CorpusBatch.ReduceSrcTokensToSingleGroup();
            if (sharedVocab)
            {
                CorpusBatch.MergeTokensCountSrcTgt(0, 0);
            }

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(srcVocabSize, tgtVocabSize, minFreq);           
            return (srcVocabs[0], tgtVocabs[0]);         
        }
    }
}
