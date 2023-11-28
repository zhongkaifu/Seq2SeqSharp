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
    public class SeqCorpus : MonoCorpus<SeqCorpusBatch>
    {

        public SeqCorpus(string corpusFilePath, string tgtLangName, int maxTokenSizePerBatch, int maxTgtSentLength = 32, PaddingEnums paddingEnums = PaddingEnums.AllowPadding, TooLongSequence tooLongSequence = TooLongSequence.Ignore, string indexedFilePath = "", int startBatchId = 0)
            : base(corpusFilePath, tgtLangName, maxTokenSizePerBatch, maxTgtSentLength, paddingEnums: paddingEnums, tooLongSequence: tooLongSequence, indexedFilePath: indexedFilePath, startBatchId: startBatchId)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public Vocab BuildVocabs(int tgtVocabSize = 45000, int minFreq = 1)
        {
            CorpusBatch.t_ds = CountTokenFreqs();
            CorpusBatch.ReduceSrcTokensToSingleGroup();

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(0, tgtVocabSize, minFreq);
            return tgtVocabs[0];
        }
    }
}
