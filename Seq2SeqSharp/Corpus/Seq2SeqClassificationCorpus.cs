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
    /// <summary>
    /// Data format:
    /// Source side: tokens split by space
    /// Target side: [classification tag] \t tokens split by space
    /// </summary>
    public class Seq2SeqClassificationCorpus : ParallelCorpus<Seq2SeqClassificationCorpusBatch>
    {

        public Seq2SeqClassificationCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
            : base(corpusFilePath, srcLangName, tgtLangName, batchSize, shuffleBlockSize, maxSrcSentLength, maxTgtSentLength, shuffleEnums: shuffleEnums)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// For return vocabs: (source vocab, target vocab, classification vocab)
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab, Vocab) BuildVocabs(int vocabSize = 45000, bool sharedVocab = false)
        {
            List<SntPair> sntPairs = new List<SntPair>();
            foreach (var sntPairBatch in this)
            {
                sntPairs.AddRange(sntPairBatch.SntPairs);
            }


            Dictionary<int, int> sharedSrcTgtVocabGroupMapping = new Dictionary<int, int>();
            if (sharedVocab)
            {
                sharedSrcTgtVocabGroupMapping.Add(0, 1); //The second column in target side is text the model will generate
            }

            (var srcVocabs, var tgtVocabs) = CorpusBatch.BuildVocabs(sntPairs, vocabSize, sharedSrcTgtVocabGroupMapping);

            Vocab srcVocab = srcVocabs[0];
            for (int i = 1; i < srcVocabs.Count; i++)
            {
                Logger.WriteLine($"Merge source vocabualry from group '{i}' to group '0'");
                srcVocab.MergeVocab(srcVocabs[i]);
            }

            Vocab clsVocab = tgtVocabs[0];
            Vocab tgtVocab = tgtVocabs[1];

            return (srcVocab, tgtVocab, clsVocab);
        }
    }
}
