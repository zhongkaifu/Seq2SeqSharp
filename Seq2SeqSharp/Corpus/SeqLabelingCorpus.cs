using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    public class SeqLabelingCorpus : ParallelCorpus<SeqLabelingCorpusBatch>
    {
        private static (string, string) ConvertSequenceLabelingFormatToParallel(string filePath)
        {
            List<string> srcLines = new List<string>();
            List<string> tgtLines = new List<string>();

            List<string> currSrcLine = new List<string>();
            List<string> currTgtLine = new List<string>();
            foreach (var line in File.ReadAllLines(filePath))
            {
                if (line.IsNullOrEmpty() )
                {
                    //This is a new record

                    srcLines.Add(string.Join(" ", currSrcLine));
                    tgtLines.Add(string.Join(" ", currTgtLine));

                    currSrcLine = new List<string>();
                    currTgtLine = new List<string>();
                }
                else
                {
                    string[] items = line.Split(new char[] { ' ', '\t' });
                    string srcItem = items[0];
                    string tgtItem = items[1];

                    currSrcLine.Add(srcItem);
                    currTgtLine.Add(tgtItem);
                }
            }

            srcLines.Add(string.Join(" ", currSrcLine));
            tgtLines.Add(string.Join(" ", currTgtLine));

            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            File.WriteAllLines(srcFilePath, srcLines);
            File.WriteAllLines(tgtFilePath, tgtLines);

            Logger.WriteLine($"Convert sequence labeling corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }


        public SeqLabelingCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSentLength;
            m_maxTgtSentLength = maxSentLength;
            m_shuffleEnums = shuffleEnums;

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();


            (string srcFilePath, string tgtFilePath) = ConvertSequenceLabelingFormatToParallel(corpusFilePath);

            m_srcFileList.Add(srcFilePath);
            m_tgtFileList.Add(tgtFilePath);
        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        public (Vocab, Vocab) BuildVocabs(int vocabSize = 45000)
        {
            foreach (var sntPairBatch in this)
            {
                CorpusBatch.CountSntPairTokens(sntPairBatch.SntPairs);
            }
            CorpusBatch.ReduceSrcTokensToSingleGroup();

            (List<Vocab> srcVocabs, List<Vocab> tgtVocabs) = CorpusBatch.GenerateVocabs(vocabSize);
            return (srcVocabs[0], tgtVocabs[0]);
        }
    }
}
