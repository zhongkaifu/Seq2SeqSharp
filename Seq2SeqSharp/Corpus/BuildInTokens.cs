using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public static class BuildInTokens
    {
        public const string EOS = "</s>";
        public const string BOS = "<s>";
        public const string UNK = "<unk>";
        public const string SEP = "[SEP]";
        public const string CLS = "[CLS]";

        public static bool IsPreDefinedToken(string str)
        {
            return str == EOS || str == BOS || str == UNK || str == CLS;
        }

        /// <summary>
        /// Pad given sentences to the same length and return their original length
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static float[] PadSentences(List<List<string>> s, int maxLen = -1)
        {
            float[] originalLengths = new float[s.Count];

            if (maxLen <= 0)
            {
                foreach (List<string> item in s)
                {
                    if (item.Count > maxLen)
                    {
                        maxLen = item.Count;
                    }
                }
            }

            for (int i = 0; i < s.Count; i++)
            {
                int count = s[i].Count;
                originalLengths[i] = count;

                for (int j = 0; j < maxLen - count; j++)
                {
                    s[i].Add(EOS);
                }
            }

            return originalLengths;
        }


        /// <summary>
        /// Pad given sentences to the same length and return their original length
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static float[] PadSentences(List<List<int>> s, int tokenToPad, int maxLen = -1)
        {
            float[] originalLengths = new float[s.Count];

            if (maxLen <= 0)
            {
                foreach (List<int> item in s)
                {
                    if (item.Count > maxLen)
                    {
                        maxLen = item.Count;
                    }
                }
            }

            for (int i = 0; i < s.Count; i++)
            {
                int count = s[i].Count;
                originalLengths[i] = count;

                for (int j = 0; j < maxLen - count; j++)
                {
                    s[i].Add(tokenToPad);
                }
            }

            return originalLengths;
        }

        public static List<List<string>> LeftShiftSnts(List<List<string>> input, string lastTokenToPad)
        {
            List<List<string>> r = new List<List<string>>();

            foreach (var seq in input)
            {
                List<string> rseq = new List<string>();

                rseq.AddRange(seq);
                rseq.RemoveAt(0);
                rseq.Add(lastTokenToPad);

                r.Add(rseq);
            }

            return r;
        }
    }
}
