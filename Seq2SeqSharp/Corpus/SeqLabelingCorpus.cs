// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Utils;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp.Tools
{
    public class SeqLabelingCorpus : ParallelCorpus<SeqLabelingCorpusBatch>
    {
        private static (string, string) ConvertSequenceLabelingFormatToParallel(string filePath)
        {
            string srcFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_src.tmp");
            string tgtFilePath = Path.Combine(Directory.GetCurrentDirectory(), Path.GetRandomFileName() + "_tgt.tmp");

            StreamWriter swSrc = new StreamWriter(srcFilePath);
            StreamWriter swTgt = new StreamWriter(tgtFilePath);

            List<string> currSrcLine = new List<string>();
            List<string> currTgtLine = new List<string>();
            foreach (var line in File.ReadLines(filePath))
            {
                if (line.IsNullOrEmpty() )
                {
                    //This is a new record

                    swSrc.WriteLine(string.Join(" ", currSrcLine));
                    swTgt.WriteLine(string.Join(" ", currTgtLine));

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

            if (currSrcLine.Count > 0)
            {
                swSrc.WriteLine(string.Join(" ", currSrcLine));
                swTgt.WriteLine(string.Join(" ", currTgtLine));
            }

            swSrc.Close();
            swTgt.Close();

            Logger.WriteLine($"Convert sequence labeling corpus file '{filePath}' to parallel corpus files '{srcFilePath}' and '{tgtFilePath}'");

            return (srcFilePath, tgtFilePath);
        }


        public SeqLabelingCorpus(string corpusFilePath, int batchSize, int maxSentLength = 128, PaddingEnums paddingEnums = PaddingEnums.AllowPadding)
        {
            Logger.WriteLine($"Loading sequence labeling corpus from '{corpusFilePath}' MaxSentLength = '{maxSentLength}'");
            m_maxTokenSizePerBatch = batchSize;
            m_maxSrcTokenSize = maxSentLength;
            m_maxTgtTokenSize = maxSentLength;
            m_paddingEnums = paddingEnums;
            CorpusName = corpusFilePath;

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
            (CorpusBatch.s_ds, CorpusBatch.t_ds) = CountTokenFreqs();
            CorpusBatch.ReduceSrcTokensToSingleGroup();

            (List<Vocab> srcVocabs, List<Vocab> tgtVocabs) = CorpusBatch.GenerateVocabs(vocabSize);
            return (srcVocabs[0], tgtVocabs[0]);
        }
    }
}
