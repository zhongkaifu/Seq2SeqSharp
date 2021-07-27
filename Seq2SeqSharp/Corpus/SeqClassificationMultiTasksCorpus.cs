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
            List<SntPair> sntPairs = new List<SntPair>();
            foreach (var sntPairBatch in this)
            {
                sntPairs.AddRange(sntPairBatch.SntPairs);
            }

            (var srcVocabs, var tgtVocabs) = CorpusBatch.BuildVocabs(sntPairs, vocabSize);
            Vocab srcVocab = srcVocabs[0];
            for (int i = 1; i < srcVocabs.Count; i++)
            {
                Logger.WriteLine($"Merge source vocabualry from group '{i}' to group '0'");
                srcVocab.MergeVocab(srcVocabs[i]);
            }

            return (srcVocab, tgtVocabs);
        }
    }
}
