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
        
        private (string, string) ConvertSequenceClassificationFormatToParallel(string filePath)
        {
            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            StreamWriter swSrc = new StreamWriter(srcFilePath);
            StreamWriter swTgt = new StreamWriter(tgtFilePath);

            foreach (var line in File.ReadAllLines(filePath))
            {
                // Format: [Category Tag1] \t [Category Tag2] \t ... \t [Category TagN] \t [Sequence Text]
                string[] items = line.Split(new char[] { '\t' });
                string srcItem = items[items.Length - 1];
                string tgtItem = String.Join('\t', items, 0, items.Length - 1).Replace(" ", "_").Replace("\t"," ");

                swSrc.WriteLine(srcItem);
                swTgt.WriteLine(tgtItem);
            }

            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Convert sequence classification corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }

        public SeqClassificationMultiTasksCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSentLength;
            m_maxTgtSentLength = maxSentLength;
            m_shuffleEnums = shuffleEnums;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();


            (string srcFilePath, string tgtFilePath) = ConvertSequenceClassificationFormatToParallel(corpusFilePath);

            m_srcFileList.Add(srcFilePath);
            m_tgtFileList.Add(tgtFilePath);
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


            return (srcVocabs[0], tgtVocabs);
        }
    }
}
