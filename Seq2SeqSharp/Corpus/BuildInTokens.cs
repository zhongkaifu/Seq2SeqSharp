// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;

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
            return str == EOS || str == BOS || str == UNK || str == CLS || str == SEP;
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
        public static float[] PadSentences(List<List<int>> s, int tokenToPad, int maxLen = -1, int alignmentFactor = 0)
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

            if (alignmentFactor > 0 && maxLen % alignmentFactor != 0)
            {
                int additionalPaddingSize = alignmentFactor - (maxLen % alignmentFactor);
                maxLen += additionalPaddingSize;
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
