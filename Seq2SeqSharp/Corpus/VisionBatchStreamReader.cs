using System;
using System.Collections.Generic;
using System.IO;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Corpus
{
    public class VisionBatchStreamReader<T> : IBatchStreamReader<T> where T : IVisionSntPairBatch, new()
    {
        private static readonly object locker = new object();
        private readonly IEnumerator<string> m_reader;
        private readonly int m_batchSize;
        private int m_currentIdx;

        public VisionBatchStreamReader(string filePath, int batchSize)
        {
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentNullException(nameof(filePath));
            }

            m_reader = File.ReadLines(filePath).GetEnumerator();
            m_batchSize = batchSize;
            m_currentIdx = 0;
        }

        public (int, IPairBatch) GetNextBatch()
        {
            lock (locker)
            {
                List<IPair> pairs = new List<IPair>();
                int startIdx = m_currentIdx;

                while (pairs.Count < m_batchSize && m_reader.MoveNext())
                {
                    string line = m_reader.Current?.Trim();
                    if (string.IsNullOrEmpty(line))
                    {
                        continue;
                    }

                    string imagePath = line;
                    string caption = string.Empty;
                    int tabIdx = line.IndexOf('\t');
                    if (tabIdx >= 0)
                    {
                        imagePath = line.Substring(0, tabIdx).Trim();
                        caption = line[(tabIdx + 1)..].Trim();
                    }

                    pairs.Add(new VisionSntPair(imagePath, caption));
                    m_currentIdx++;
                }

                if (pairs.Count == 0)
                {
                    return (-1, null);
                }

                T batch = new T();
                batch.CreateBatch(pairs);
                return (startIdx, batch);
            }
        }
    }
}
