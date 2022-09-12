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
using System.Collections.Generic;

namespace Seq2SeqSharp.Corpus
{

    public class SeqClassificationMultiTasksCorpus : ParallelCorpus<SeqClassificationMultiTasksCorpusBatch>
    {        
        public SeqClassificationMultiTasksCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int maxTokenSizePerBatch, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
            : base(corpusFilePath, srcLangName, tgtLangName, maxTokenSizePerBatch, maxSentLength, maxSentLength, shuffleEnums: shuffleEnums, tooLongSequence: tooLongSequence)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, List<Vocab>) BuildVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000)
        {
            (CorpusBatch.s_ds, CorpusBatch.t_ds) = CountTokenFreqs();

            CorpusBatch.ReduceSrcTokensToSingleGroup();

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(srcVocabSize, tgtVocabSize);
            return (srcVocabs[0], tgtVocabs);
        }
    }
}
