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
    public class SequenceClassificationCorpus : ParallelCorpus
    {
        private (string, string) ConvertSequenceClassificationFormatToParallel(string filePath)
        {
            List<string> srcLines = new List<string>();
            List<string> tgtLines = new List<string>();

            foreach (var line in File.ReadAllLines(filePath))
            {
                // Format: [Category Tag] \t [Sequence Text]
                string[] items = line.Split(new char[] { '\t' });
                string srcItem = items[1];
                string tgtItem = items[0];

                srcLines.Add(srcItem);
                tgtLines.Add(tgtItem);

            }

            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            File.WriteAllLines(srcFilePath, srcLines);
            File.WriteAllLines(tgtFilePath, tgtLines);

            Logger.WriteLine($"Convert sequence classification corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }

        public SequenceClassificationCorpus(string corpusFilePath, int batchSize, int shuffleBlockSize = -1, int maxSentLength = 128, ShuffleEnums shuffleEnums = ShuffleEnums.Random)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_batchSize = batchSize;
            m_blockSize = shuffleBlockSize;
            m_maxSrcSentLength = maxSentLength;
            m_maxTgtSentLength = maxSentLength;
            m_shuffleEnums = shuffleEnums;
            m_sentSrcPrefix = ParallelCorpus.CLS;
            m_sentSrcSuffifx = "";
            m_sentTgtPrefix = "";
            m_sentTgtSuffix = "";

            m_srcFileList = new List<string>();
            m_tgtFileList = new List<string>();


            (string srcFilePath, string tgtFilePath) = ConvertSequenceClassificationFormatToParallel(corpusFilePath);

            m_srcFileList.Add(srcFilePath);
            m_tgtFileList.Add(tgtFilePath);
        }
    }
}
