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
    public class SntBatchStreamReader<T> : IBatchStreamReader<T> where T : IPairBatch, new()
    {
        static object locker = new object();

        int currentIdx;
        int maxSentLength;
        int batchSize;
        SentencePiece sp = null;
        IEnumerator<string> reader = null;

        public SntBatchStreamReader(string filePath, int batchSize, int maxSentLength, string sentencePieceModelPath = null)
        {
            currentIdx = 0;
            this.maxSentLength = maxSentLength;
            this.batchSize = batchSize;
            reader = File.ReadLines(filePath).GetEnumerator();


            if (String.IsNullOrEmpty(sentencePieceModelPath) == false)
            {
                Logger.WriteLine($"Loading sentence piece model '{sentencePieceModelPath}' for encoding.");
                sp = new SentencePiece(sentencePieceModelPath);
            }
        }


        public (int, IPairBatch) GetNextBatch()
        {
            List<List<string>> inputBatchs = new List<List<string>>(); // shape: [batch_size, sequence_length]

            lock (locker)
            {
                int oldIdx = currentIdx;

                for (int i = 0; i < batchSize && reader.MoveNext(); i++, currentIdx++)
                {
                    string line = reader.Current;                                    
                    if (sp != null)
                    {
                        line = sp.Encode(line);
                    }
                    Misc.AppendNewBatch(inputBatchs, line, maxSentLength);
                }

                if (inputBatchs.Count == 0)
                {
                    return (-1, null);
                }

                T batch = new T();
                batch.CreateBatch(inputBatchs, null);

                return (oldIdx, batch);
            }
        }
    }
}
