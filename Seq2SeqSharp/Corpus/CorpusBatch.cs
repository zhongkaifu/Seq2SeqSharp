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
        public List<List<List<string>>> SrcTknsGroups = null; // shape (group_size, batch_size, seq_size)
        public List<List<List<string>>> TgtTknsGroups = null;


        public List<SntPair> SntPairs;

        public int BatchSize => SrcTknsGroups[0].Count;

        public int SrcTokenCount { get; set; }
        public int TgtTokenCount { get; set; }

        public virtual ISntPairBatch CloneSrcTokens()
        {
            throw new NotImplementedException();
        }

        public static void TryAddPrefix(List<List<string>> tokens, string prefix)
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


        public static void TryAddSuffix(List<List<string>> tokens, string suffix)
        {
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i].Count == 0)
                {
                    tokens[i].Add(suffix);
                }
                else
                {
                    if (tokens[i][^1] != suffix)
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

            SrcTknsGroups = new List<List<List<string>>>();
            TgtTknsGroups = new List<List<List<string>>>();


            for (int i = 0; i < sntPairs[0].SrcTokenGroups.Count; i++)
            {
                SrcTknsGroups.Add(new List<List<string>>());
            }

            int srcTknsGroupNum = SrcTknsGroups.Count;

            for (int i = 0; i < sntPairs[0].TgtTokenGroups.Count; i++)
            {
                TgtTknsGroups.Add(new List<List<string>>());
            }

            int tgtTknsGroupNum = TgtTknsGroups.Count;

            for (int i = 0; i < sntPairs.Count; i++)
            {
                if (sntPairs[i].SrcTokenGroups.Count != srcTknsGroupNum)
                {
                    throw new DataMisalignedException($"Source data '{i}' group size is mismatch. It's {sntPairs[i].SrcTokenGroups.Count}, but it should be {srcTknsGroupNum}. Tokens: {sntPairs[i].PrintSrcTokens()}");
                }

                for (int j = 0; j < srcTknsGroupNum; j++)
                {
                    SrcTknsGroups[j].Add(sntPairs[i].SrcTokenGroups[j]);
                    SrcTokenCount += sntPairs[i].SrcTokenGroups[j].Count;

                }

                if (sntPairs[i].TgtTokenGroups.Count != tgtTknsGroupNum)
                {
                    throw new DataMisalignedException($"Target data '{i}' group size is mismatch. It's {sntPairs[i].TgtTokenGroups.Count}, but it should be {tgtTknsGroupNum}. Tokens: {sntPairs[i].PrintTgtTokens()}");
                }

                for (int j = 0; j < tgtTknsGroupNum; j++)
                {
                    TgtTknsGroups[j].Add(sntPairs[i].TgtTokenGroups[j]);
                    TgtTokenCount += sntPairs[i].TgtTokenGroups[j].Count;
                }
            }
        }

        public ISntPairBatch GetRange(int idx, int count)
        {
            CorpusBatch cb = new CorpusBatch
            {
                SrcTknsGroups = new List<List<List<string>>>()
            };
            for (int i = 0; i < SrcTknsGroups.Count; i++)
            {
                cb.SrcTknsGroups.Add(new List<List<string>>());
                cb.SrcTknsGroups[i].AddRange(SrcTknsGroups[i].GetRange(idx, count));
            }

            if (TgtTknsGroups != null)
            {
                cb.TgtTknsGroups = new List<List<List<string>>>();
                for (int i = 0; i < TgtTknsGroups.Count; i++)
                {
                    cb.TgtTknsGroups.Add(new List<List<string>>());

                    if (TgtTknsGroups[i].Count > 0)
                    {
                        cb.TgtTknsGroups[i].AddRange(TgtTknsGroups[i].GetRange(idx, count));
                    }
                }
            }
            else
            {
                cb.TgtTknsGroups = TgtTknsGroups;
            }


            return cb;
        }

        public List<List<string>> GetSrcTokens(int group)
        {
            return SrcTknsGroups[group];
        }

        public List<List<string>> GetTgtTokens(int group)
        {
            return TgtTknsGroups[group];
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



        // count up all words
        static readonly List<Dictionary<string, int>> s_ds = new List<Dictionary<string, int>>();
        static readonly List<Dictionary<string, int>> t_ds = new List<Dictionary<string, int>>();



        static public void MergeTokensCountSrcTgt(int srcGroupIdx, int tgtGroupIdx)
        {
            Logger.WriteLine($"Merge tokens from source group '{srcGroupIdx}' to target group '{tgtGroupIdx}'");
            foreach (var pair in t_ds[tgtGroupIdx])
            {
                if (s_ds[srcGroupIdx].ContainsKey(pair.Key))
                {
                    s_ds[srcGroupIdx][pair.Key] += pair.Value;
                }
                else
                {
                    s_ds[srcGroupIdx].Add(pair.Key, pair.Value);
                }
            }

            t_ds[tgtGroupIdx] = s_ds[srcGroupIdx];

        }

        static public void ReduceSrcTokensToSingleGroup()
        {
            Logger.WriteLine($"Reduce source vocabs group from '{s_ds.Count}' to 1");
            Dictionary<string, int> rst = new Dictionary<string, int>();

            foreach (var dict in s_ds)
            {
                foreach (var pair in dict)
                {
                    if (rst.ContainsKey(pair.Key))
                    {
                        rst[pair.Key] += pair.Value;
                    }
                    else
                    {
                        rst.Add(pair.Key, pair.Value);
                    }

                }
            }

            s_ds.Clear();
            s_ds.Add(rst);
        }


        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        /// side and the second group in target side are shared vocabulary</param>
        static public void CountSntPairTokens(List<SntPair> sntPairs)
        {
            foreach (SntPair sntPair in sntPairs)
            {
                if (s_ds.Count == 0)
                {
                    for (int i = 0; i < sntPair.SrcTokenGroups.Count; i++)
                    {
                        s_ds.Add(new Dictionary<string, int>());
                    }
                }

                if (t_ds.Count == 0)
                {
                    for (int i = 0; i < sntPair.TgtTokenGroups.Count; i++)
                    {
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
                    }

                }

            }            
        }


        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        /// <param name="sharedSrcTgtVocabGroupMapping">The mappings for shared vocabularies between source side and target side. The values in the mappings are group ids. For example: sharedSrcTgtVocabGroupMapping[0] = 1 means the first group in source
        /// side and the second group in target side are shared vocabulary</param>
        static public (List<Vocab>, List<Vocab>) GenerateVocabs(int srcVocabSize = 45000, int tgtVocabSize = 45000)
        {
            Logger.WriteLine($"Building vocabulary from corpus.");

            List<Vocab> srcVocabs = InnerBuildVocab(srcVocabSize, s_ds, "Source");
            List<Vocab> tgtVocabs = InnerBuildVocab(tgtVocabSize, t_ds, "Target");

            s_ds.Clear();
            t_ds.Clear();

            return (srcVocabs, tgtVocabs);
        }

        private static List<Vocab> InnerBuildVocab(int vocabSize, List<Dictionary<string, int>> ds, string tag)
        {
            List<Vocab> vocabs = new List<Vocab>();

            for (int i = 0; i < ds.Count; i++)
            {
                Vocab vocab = new Vocab();
                SortedDictionary<int, List<string>> sd = new SortedDictionary<int, List<string>>();

                var s_d = ds[i];
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
                            vocab.WordToIndex[token] = q;
                            vocab.IndexToWord[q] = token;
                            vocab.Items.Add(token);
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

                vocabs.Add(vocab);

                Logger.WriteLine($"{tag} Vocab Group '{i}': Original vocabulary size = '{s_d.Count}', Truncated vocabulary size = '{q}'");

            }

            return vocabs;
        }
    
        public int GetSrcGroupSize()
        {
            return SrcTknsGroups.Count;
        }

        public int GetTgtGroupSize()
        {
            return TgtTknsGroups.Count;
        }

        public virtual void CreateBatch(List<List<List<string>>> srcTokensGroups)
        {
            throw new NotImplementedException();
        }
    }
}
