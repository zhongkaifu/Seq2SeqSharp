using AdvUtils;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class CorpusBatch : ISntPairBatch
    {
        public List<List<List<string>>> SrcTknsGroup = null; // shape (group_size, batch_size, seq_size)
        public List<List<List<string>>> TgtTknsGroup = null;


        public List<SntPair> SntPairs;

        public int BatchSize => SntPairs.Count;

        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public virtual ISntPairBatch CloneSrcTokens()
        {
            throw new NotImplementedException();
        }


        public void TryAddPrefix(List<List<string>> tokens, string prefix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(prefix);
                }
                else
                {
                    if (tokens[i][0] != prefix)
                    {
                        tokens[i].Insert(0, prefix);
                    }
                }
            }
        }


        public void TryAddSuffix(List<List<string>> tokens, string suffix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(suffix);
                }
                else
                {
                    if (tokens[i][tokens[i].Count - 1] != suffix)
                    {
                        tokens[i].Add(suffix);
                    }
                }
            }
        }

        public virtual void CreateBatch(List<SntPair> sntPairs)
        {
            SrcTokenCount = 0;
            TgtTokenCount = 0;


            SntPairs = sntPairs;

            SrcTknsGroup = new List<List<List<string>>>();
            TgtTknsGroup = new List<List<List<string>>>();


            for (int i = 0; i < sntPairs[0].SrcTokenGroups.Count; i++)
            {
                SrcTknsGroup.Add(new List<List<string>>());
            }

            int srcTknsGroupNum = SrcTknsGroup.Count;

            for (int i = 0; i < sntPairs[0].TgtTokenGroups.Count; i++)
            {
                TgtTknsGroup.Add(new List<List<string>>());
            }

            int tgtTknsGroupNum = TgtTknsGroup.Count;

            for (int i = 0; i < sntPairs.Count; i++)
            {
                for (int j = 0; j < srcTknsGroupNum; j++)
                {
                    SrcTknsGroup[j].Add(sntPairs[i].SrcTokenGroups[j]);
                    SrcTokenCount += sntPairs[i].SrcTokenGroups[j].Count;

                }


                for (int j = 0; j < tgtTknsGroupNum; j++)
                {
                    TgtTknsGroup[j].Add(sntPairs[i].TgtTokenGroups[j]);
                    TgtTokenCount += sntPairs[i].TgtTokenGroups[j].Count;
                }
            }
        }

        public ISntPairBatch GetRange(int idx, int count)
        {
            CorpusBatch cb = new CorpusBatch();

            cb.SrcTknsGroup = new List<List<List<string>>>();
            for (int i = 0; i < SrcTknsGroup.Count; i++)
            {
                cb.SrcTknsGroup.Add(new List<List<string>>());
                cb.SrcTknsGroup[i].AddRange(SrcTknsGroup[i].GetRange(idx, count));
            }

            if (TgtTknsGroup != null)
            {
                cb.TgtTknsGroup = new List<List<List<string>>>();
                for (int i = 0; i < TgtTknsGroup.Count; i++)
                {
                    cb.TgtTknsGroup.Add(new List<List<string>>());
                    cb.TgtTknsGroup[i].AddRange(TgtTknsGroup[i].GetRange(idx, count));
                }
            }
            else
            {
                cb.TgtTknsGroup = null;
            }

            return cb;
        }

        public List<List<string>> GetSrcTokens(int group)
        {
            return SrcTknsGroup[group];
        }

        public List<List<string>> GetTgtTokens(int group)
        {
            return TgtTknsGroup[group];
        }

        public List<List<string>> InitializeHypTokens(string prefix)
        {
            List<List<string>> hypTkns = new List<List<string>>();
            for (int i = 0; i < BatchSize; i++)
            {
                if (String.IsNullOrEmpty(prefix) == false)
                {
                    hypTkns.Add(new List<string>() { prefix });
                }
                else
                {
                    hypTkns.Add(new List<string>());
                }
            }

            return hypTkns;
        }




        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        /// <param name="sharedSrcTgtVocabGroupMapping">The mappings for shared vocabularies between source side and target side. The values in the mappings are group ids. For example: sharedSrcTgtVocabGroupMapping[0] = 1 means the first group in source
        /// side and the second group in target side are shared vocabulary</param>
        static public (List<Vocab>, List<Vocab>) BuildVocabs(List<SntPair> sntPairs, int vocabSize = 45000, Dictionary<int, int> sharedSrcTgtVocabGroupMapping = null)
        {
            List<Vocab> srcVocabs = new List<Vocab>();
            List<Vocab> tgtVocabs = new List<Vocab>();

            Logger.WriteLine($"Building vocabulary from corpus.");

            // count up all words
            List<Dictionary<string, int>> s_ds = new List<Dictionary<string, int>>();
            List<Dictionary<string, int>> t_ds = new List<Dictionary<string, int>>();

            Dictionary<int, int> sharedTgtSrcVocabGroupMapping = null;
            if (sharedSrcTgtVocabGroupMapping != null)
            {
                sharedTgtSrcVocabGroupMapping = new Dictionary<int, int>();
                foreach (var pair in sharedSrcTgtVocabGroupMapping)
                {
                    sharedTgtSrcVocabGroupMapping.Add(pair.Value, pair.Key);
                }
            }



            foreach (SntPair sntPair in sntPairs)
            {
                if (srcVocabs.Count == 0)
                {
                    for (int i = 0; i < sntPair.SrcTokenGroups.Count; i++)
                    {
                        srcVocabs.Add(new Vocab());
                        s_ds.Add(new Dictionary<string, int>());
                    }
                }

                if (tgtVocabs.Count == 0)
                {
                    for (int i = 0; i < sntPair.TgtTokenGroups.Count; i++)
                    {
                        tgtVocabs.Add(new Vocab());
                        t_ds.Add(new Dictionary<string, int>());
                    }
                }


                for (int g = 0; g < sntPair.SrcTokenGroups.Count; g++)
                {
                    var tokens = sntPair.SrcTokenGroups[g];
                    for (int i = 0; i < tokens.Count; i++)
                    {
                        var token = tokens[i];
                        if (s_ds[g].ContainsKey(token) == true)
                        {
                            s_ds[g][token]++;
                        }
                        else
                        {
                            s_ds[g].Add(token, 1);
                        }

                        if (sharedSrcTgtVocabGroupMapping != null && sharedSrcTgtVocabGroupMapping.ContainsKey(g))
                        {
                            var mappedTgtGroup = sharedSrcTgtVocabGroupMapping[g];

                            if (t_ds[mappedTgtGroup].ContainsKey(token) == true)
                            {
                                t_ds[mappedTgtGroup][token]++;
                            }
                            else
                            {
                                t_ds[mappedTgtGroup].Add(token, 1);
                            }

                        }
                    }

                }


                for (int g = 0; g < sntPair.TgtTokenGroups.Count; g++)
                {
                    var tokens = sntPair.TgtTokenGroups[g];
                    for (int i = 0; i < tokens.Count; i++)
                    {
                        var token = tokens[i];
                        if (t_ds[g].ContainsKey(token) == true)
                        {
                            t_ds[g][token]++;
                        }
                        else
                        {
                            t_ds[g].Add(token, 1);
                        }

                        if (sharedTgtSrcVocabGroupMapping != null && sharedTgtSrcVocabGroupMapping.ContainsKey(g))                   
                        {
                            var mappedSrcGroup = sharedTgtSrcVocabGroupMapping[g];
                            if (s_ds[mappedSrcGroup].ContainsKey(token) == true)
                            {
                                s_ds[mappedSrcGroup][token]++;
                            }
                            else
                            {
                                s_ds[mappedSrcGroup].Add(token, 1);
                            }

                        }
                    }

                }

            }


            for (int i = 0; i < s_ds.Count; i++)
            {
                SortedDictionary<int, List<string>> sd = new SortedDictionary<int, List<string>>();

                var s_d = s_ds[i];
                foreach (var kv in s_d)
                {
                    if (sd.ContainsKey(kv.Value) == false)
                    {
                        sd.Add(kv.Value, new List<string>());
                    }
                    sd[kv.Value].Add(kv.Key);
                }

                int q = 3;
                foreach (var kv in sd.Reverse())
                {
                    foreach (var token in kv.Value)
                    {
                        if (BuildInTokens.IsPreDefinedToken(token) == false)
                        {
                            // add word to vocab
                            srcVocabs[i].WordToIndex[token] = q;
                            srcVocabs[i].IndexToWord[q] = token;
                            srcVocabs[i].Items.Add(token);
                            q++;

                            if (q >= vocabSize)
                            {
                                break;
                            }
                        }
                    }

                    if (q >= vocabSize)
                    {
                        break;
                    }
                }

                Logger.WriteLine($"Source Vocab Group '{i}': Original vocabulary size = '{s_d.Count}', Truncated vocabulary size = '{q}'");

            }


            for (int i = 0; i < t_ds.Count; i++)
            {
                SortedDictionary<int, List<string>> sd = new SortedDictionary<int, List<string>>();

                var t_d = t_ds[i];
                foreach (var kv in t_d)
                {
                    if (sd.ContainsKey(kv.Value) == false)
                    {
                        sd.Add(kv.Value, new List<string>());
                    }
                    sd[kv.Value].Add(kv.Key);
                }

                int q = 3;
                foreach (var kv in sd.Reverse())
                {
                    foreach (var token in kv.Value)
                    {
                        if (BuildInTokens.IsPreDefinedToken(token) == false)
                        {
                            // add word to vocab
                            tgtVocabs[i].WordToIndex[token] = q;
                            tgtVocabs[i].IndexToWord[q] = token;
                            tgtVocabs[i].Items.Add(token);
                            q++;

                            if (q >= vocabSize)
                            {
                                break;
                            }
                        }
                    }

                    if (q >= vocabSize)
                    {
                        break;
                    }
                }

                Logger.WriteLine($"Target Vocab Group '{i}': Original vocabulary size = '{t_d.Count}', Truncated vocabulary size = '{q}'");

            }


            return (srcVocabs, tgtVocabs);
        }

    }
}
