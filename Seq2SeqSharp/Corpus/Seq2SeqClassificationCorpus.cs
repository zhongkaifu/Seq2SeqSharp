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
        public (Vocab, Vocab, Vocab) BuildVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000, bool sharedVocab = false)
        {
            if (sharedVocab && (srcVocabSize != tgtVocabSize))
            {
                throw new ArgumentException($"Vocab size must be equal if sharedVocab is true. Src Vocab Size = '{srcVocabSize}', Tgt Vocab Size = '{tgtVocabSize}'");
            }

            foreach (var sntPairBatch in this)
            {
                CorpusBatch.CountSntPairTokens(sntPairBatch.SntPairs);
            }

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
