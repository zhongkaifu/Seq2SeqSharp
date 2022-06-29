// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp._SentencePiece;
using AdvUtils;

namespace Seq2SeqSharp.Corpus
{
    public class SntPairBatchStreamReader<T> : IBatchStreamReader<T> where T : ISntPairBatch, new()
    {
        static object locker = new object();

        int currentIdx;
        int maxSentLength;
        int batchSize;
        IEnumerator<string> srcReader = null;
        IEnumerator<string> tgtReader = null;

        SentencePiece srcSP = null;
        SentencePiece tgtSP = null;

        public SntPairBatchStreamReader(string srcFilePath, string tgtFilePath, int batchSize, int maxSentLength, string srcSPMPath = null, string tgtSPMPath = null)
        {
            currentIdx = 0;
            this.maxSentLength = maxSentLength;
            this.batchSize = batchSize;

            Logger.WriteLine($"Loading lines from '{srcFilePath}'");
            srcReader = File.ReadLines(srcFilePath).GetEnumerator();

            Logger.WriteLine($"Loading lines from '{tgtFilePath}'");
            tgtReader = File.ReadLines(tgtFilePath).GetEnumerator();

            if (String.IsNullOrEmpty(srcSPMPath) == false)
            {
                Logger.WriteLine($"Loading sentence piece model '{srcSPMPath}' for encoding.");
                srcSP = new SentencePiece(srcSPMPath);
            }

            if (String.IsNullOrEmpty(tgtSPMPath) == false)
            {
                Logger.WriteLine($"Loading sentence piece model '{tgtSPMPath}' for encoding.");
                tgtSP = new SentencePiece(tgtSPMPath);
            }
        }


        public (int, ISntPairBatch) GetNextBatch()
        {
            List<List<List<string>>> inputBatchs = new List<List<List<string>>>(); // shape: (feature_group_size, batch_size, sequence_length)
            List<List<List<string>>> outputBatchs = new List<List<List<string>>>(); // shape: (feature_group_size, batch_size, sequence_length)

            lock (locker)
            {
                int oldIdx = currentIdx;

                for (int i = 0; i < batchSize && srcReader.MoveNext() && tgtReader.MoveNext(); i++, currentIdx++)
                {
                    string line = srcReader.Current;
                    if (srcSP != null)
                    {
                        line = srcSP.Encode(line);
                    }
                    Misc.AppendNewBatch(inputBatchs, line, maxSentLength);


                    line = tgtReader.Current;
                    if (tgtSP != null)
                    {
                        line = tgtSP.Encode(line);
                    }
                    Misc.AppendNewBatch(outputBatchs, line, maxSentLength);
                }

                if (inputBatchs.Count == 0)
                {
                    return (-1, null);
                }

                T batch = new T();
                batch.CreateBatch(inputBatchs, outputBatchs);

                return (oldIdx, batch);
            }
        }
    }
}
