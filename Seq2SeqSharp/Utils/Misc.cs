using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Utils
{
    public class RoundArray<T>
    {
        private readonly T[] m_array;
        private int currentIdx = 0;
        public RoundArray(T[] a)
        {
            m_array = a;
        }

        public T GetNextItem()
        {
            T item = m_array[currentIdx];
            currentIdx = (currentIdx + 1) % m_array.Length;

            return item;
        }
    }

    public class Utils
    {
        public static string GetTimeStamp(DateTime timeStamp)
        {
            return string.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }
    }

    public class Misc
    {
        public static void AppendNewBatch(List<List<List<string>>> inputBatchs, string line, int maxTokenLength)
        {
            string[] groups = line.Trim().Split('\t');

            if (inputBatchs.Count == 0)
            {
                for (int i = 0; i < groups.Length; i++)
                {
                    inputBatchs.Add(new List<List<string>>());
                }
            }

            for (int i = 0; i < groups.Length; i++)
            {
                var group = groups[i];
                List<string> tokens = group.Trim().Split(' ').ToList();
                if (tokens.Count > maxTokenLength - 2)
                {
                    tokens = tokens.GetRange(0, maxTokenLength - 2);
                }
                inputBatchs[i].Add(tokens);
            }
        }


    }
}
