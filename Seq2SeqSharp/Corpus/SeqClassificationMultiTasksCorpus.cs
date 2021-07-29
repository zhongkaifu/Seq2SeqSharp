using AdvUtils;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{

    public class SeqClassificationMultiTasksCorpus : ParallelCorpus<SeqClassificationMultiTasksCorpusBatch>
    {        
        public SeqClassificationMultiTasksCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
            : base(corpusFilePath, srcLangName, tgtLangName, batchSize, shuffleBlockSize, maxSentLength, maxSentLength, shuffleEnums: shuffleEnums)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, List<Vocab>) BuildVocabs(int vocabSize = 45000)
        {
            foreach (var sntPairBatch in this)
            {
                CorpusBatch.CountSntPairTokens(sntPairBatch.SntPairs);
            }

            CorpusBatch.ReduceSrcTokensToSingleGroup();

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(vocabSize);
            return (srcVocabs[0], tgtVocabs);
        }
    }
}
