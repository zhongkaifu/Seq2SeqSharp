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
    public class DPOCorpus : DPOPairCorpus<Seq2SeqCorpusBatch>
    {

        public DPOCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSrcSentLength = 32, int maxTgtSentLength = 32, PaddingEnums paddingEnums = PaddingEnums.AllowPadding, TooLongSequence tooLongSequence = TooLongSequence.Ignore, string indexedFilePath = null, int startBatchId = 0, string dataPassword = "")
            : base(corpusFilePath, srcLangName, tgtLangName, maxTokenSizePerBatch, maxSrcSentLength, maxTgtSentLength, paddingEnums: paddingEnums, tooLongSequence: tooLongSequence, indexedFilePath: indexedFilePath, startBatchId: startBatchId, dataPassword: dataPassword)
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
