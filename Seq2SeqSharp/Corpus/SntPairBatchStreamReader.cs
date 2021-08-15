using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Corpus
{
    public class SntPairBatchStreamReader<T> where T : ISntPairBatch, new()
    {
        static object locker = new object();

        int currentIdx;
        int maxSentLength;
        int batchSize;
        string[] lines;
        public SntPairBatchStreamReader(string filePath, int batchSize, int maxSentLength)
        {
            currentIdx = 0;
            this.maxSentLength = maxSentLength;
            this.batchSize = batchSize;
            lines = File.ReadAllLines(filePath);
        }


        public (int, ISntPairBatch) GetNextBatch()
        {
            List<List<List<string>>> inputBatchs = new List<List<List<string>>>(); // shape: (feature_group_size, batch_size, sequence_length)

            lock (locker)
            {
                int oldIdx = currentIdx;

                for (int i = 0; i < batchSize && currentIdx < lines.Length; i++, currentIdx++)
                {
                    Misc.AppendNewBatch(inputBatchs, lines[currentIdx], maxSentLength);
                }

                if (inputBatchs.Count == 0)
                {
                    return (-1, null);
                }

                T batch = new T();
                batch.CreateBatch(inputBatchs);

                return (oldIdx, batch);
            }
        }
    }
}
