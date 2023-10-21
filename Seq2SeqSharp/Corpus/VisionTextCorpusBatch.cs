using AdvUtils;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class VisionSntPair : IPair
    {
        public string visPath;
        public List<string> TgtTokens; //shape: [sequence_length]

        public VisionSntPair(string srcLine, string tgtLine)
        {
            visPath = srcLine;
            TgtTokens = tgtLine.Split(' ').ToList();
        }

        public int GetTgtTokenCount()
        {
            return TgtTokens.Count;
        }

        public string PrintSrcTokens()
        {
            return visPath;
        }

        public string PrintTgtTokens()
        {
            return string.Join(" ", TgtTokens.Count);
        }

    }

    public class VisionTextCorpusBatch : IVisionSntPairBatch
    {
        public List<string> SrcBatchPaths = null;
        public List<List<string>> TgtBatchTokens = null; // shape [batch_size, seq_size]

        public int BatchSize => SrcBatchPaths.Count;

        public int SrcTokenCount { get; set; } = 1;
        public int TgtTokenCount { get; set; }

        public IPairBatch CloneSrcTokens()
        {
            VisionTextCorpusBatch spb = new VisionTextCorpusBatch();
            spb.SrcBatchPaths = SrcBatchPaths;
            spb.TgtBatchTokens = InitializeHypTokens(BuildInTokens.BOS);

            return spb;
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

        public virtual void CreateBatch(List<IPair> sntPairs)
        {
            TgtTokenCount = 0;
            SrcBatchPaths = new List<string>();
            TgtBatchTokens = new List<List<string>>();

            int tgtTknsGroupNum = TgtBatchTokens.Count;

            for (int i = 0; i < sntPairs.Count; i++)
            {
                VisionSntPair pair = sntPairs[i] as VisionSntPair;

                SrcBatchPaths.Add(pair.visPath);
                TgtBatchTokens.Add(pair.TgtTokens);
                TgtTokenCount += pair.TgtTokens.Count;
            }

            TryAddPrefix(TgtBatchTokens, BuildInTokens.BOS);
            TryAddSuffix(TgtBatchTokens, BuildInTokens.EOS);
        }

        public IPairBatch GetRange(int idx, int count)
        {
            VisionTextCorpusBatch cb = new VisionTextCorpusBatch();
            cb.SrcBatchPaths = SrcBatchPaths.GetRange(idx, count);
            cb.TgtBatchTokens = TgtBatchTokens.GetRange(idx, count);

            return cb;
        }

        public List<List<string>> GetSrcTokens()
        {
            return new List<List<string>> { SrcBatchPaths };
        }

        public List<List<string>> GetTgtTokens()
        {
            return TgtBatchTokens;
        }

        public List<List<string>> InitializeHypTokens(string prefix)
        {
            List<List<string>> hypTkns = new List<List<string>>();
            for (int i = 0; i < BatchSize; i++)
            {
                if (!prefix.IsNullOrEmpty())
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
        public static List<Dictionary<string, long>> t_ds = new List<Dictionary<string, long>>();

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="vocabSize"></param>
        /// <param name="sharedSrcTgtVocabGroupMapping">The mappings for shared vocabularies between source side and target side. The values in the mappings are group ids. For example: sharedSrcTgtVocabGroupMapping[0] = 1 means the first group in source
        /// side and the second group in target side are shared vocabulary</param>
        static public List<Vocab> GenerateVocabs(int tgtVocabSize = 45000, int minFreq = 1)
        {
            Logger.WriteLine($"Building vocabulary from corpus.");

            List<Vocab> tgtVocabs = null;
            if (tgtVocabSize > 0)
            {
                tgtVocabs = InnerBuildVocab(tgtVocabSize, t_ds, "Target", minFreq);
            }
            t_ds.Clear();

            return tgtVocabs;
        }

        private static List<Vocab> InnerBuildVocab(int vocabSize, List<Dictionary<string, long>> ds, string tag, int minFreq = 1)
        {
            List<Vocab> vocabs = new List<Vocab>();

            for (int i = 0; i < ds.Count; i++)
            {
                Vocab vocab = new Vocab();
                SortedDictionary<long, List<string>> sd = new SortedDictionary<long, List<string>>();

                var s_d = ds[i];
                foreach (var kv in s_d)
                {
                    if (sd.ContainsKey(kv.Value) == false)
                    {
                        sd.Add(kv.Value, new List<string>());
                    }
                    sd[kv.Value].Add(kv.Key);
                }

                int q = vocab.IndexToWord.Count;
                foreach (var kv in sd.Reverse())
                {
                    if (kv.Key < minFreq)
                    {
                        break;
                    }

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

                if (q % 2 != 0)
                {
                    Logger.WriteLine($"Added a pad token into vocabulary for alignment.");
                    string pad = "[PAD_0]";
                    vocab.WordToIndex[pad] = q;
                    vocab.IndexToWord[q] = pad;
                    vocab.Items.Add(pad);
                    q++;
                }

                vocabs.Add(vocab);

                Logger.WriteLine($"{tag} Vocab Group '{i}': Original vocabulary size = '{s_d.Count}', Truncated vocabulary size = '{q}', Minimum Token Frequency = '{minFreq}'");

            }

            return vocabs;
        }

        public int GetSrcGroupSize()
        {
            return SrcBatchPaths.Count;
        }

        public int GetTgtGroupSize()
        {
            return TgtBatchTokens.Count;
        }

        public void CreateBatch(List<List<string>> srcTokens, List<List<string>> tgtTokens)
        {
            SrcBatchPaths = new List<string>();
            foreach (var src in srcTokens)
            {
                SrcBatchPaths.Add(src[0]);
            }

            if (tgtTokens != null)
            {
                TgtBatchTokens = tgtTokens;
                TryAddPrefix(TgtBatchTokens, BuildInTokens.BOS);
            }
            else
            {
                TgtBatchTokens = InitializeHypTokens(BuildInTokens.BOS);
            }
        }
    }
}
