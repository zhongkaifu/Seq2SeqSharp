using AdvUtils;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class Seq2SeqCorpus : ParallelCorpus<Seq2SeqCorpusBatch>
    {

        public Seq2SeqCorpus(string corpusFilePath, string srcLangName, string tgtLangName, int batchSize, int shuffleBlockSize = -1, int maxSrcSentLength = 32, int maxTgtSentLength = 32, ShuffleEnums shuffleEnums = ShuffleEnums.Random, TooLongSequence tooLongSequence = TooLongSequence.Ignore)
            :base (corpusFilePath, srcLangName, tgtLangName, batchSize, shuffleBlockSize, maxSrcSentLength, maxTgtSentLength, shuffleEnums: shuffleEnums, tooLongSequence: tooLongSequence)
        {

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab) BuildVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000, bool sharedVocab = false)
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

            (var srcVocabs, var tgtVocabs) = CorpusBatch.GenerateVocabs(srcVocabSize, tgtVocabSize);           
            return (srcVocabs[0], tgtVocabs[0]);         
        }
    }
}
