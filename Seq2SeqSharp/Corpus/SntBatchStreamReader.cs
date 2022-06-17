using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp._SentencePiece;
using AdvUtils;

namespace Seq2SeqSharp.Corpus
{
    public class SntBatchStreamReader<T> : IBatchStreamReader<T> where T : ISntPairBatch, new()
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


        public (int, ISntPairBatch) GetNextBatch()
        {
            List<List<List<string>>> inputBatchs = new List<List<List<string>>>(); // shape: (feature_group_size, batch_size, sequence_length)

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
