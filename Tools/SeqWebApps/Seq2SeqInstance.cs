// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Seq2SeqSharp;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Applications;
using AdvUtils;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Tools;

namespace Seq2SeqWebApps
{
    public enum ModelType
    { 
        EncoderDecoder,
        DecoderOnly    
    } 
    
    public static class Seq2SeqInstance
    {
        static private BaseSeq2SeqFramework<Seq2SeqModel>? m_seq2seq;
        static private SentencePiece? m_srcSpm = null;
        static private SentencePiece? m_tgtSpm = null;
        static private Seq2SeqOptions? opts;
        static Semaphore? sm = null;
        static List<int>? m_blockedTokens = null;
        static ModelType m_modelType = ModelType.EncoderDecoder;

        static public int MaxTokenToGenerate => opts.MaxTgtSentLength;
        static Dictionary<string, string> m_wordMappings = null;
        static object locker = new object();

        static public void Initialization(string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, ProcessorTypeEnums processorType, string deviceIds, SentencePiece? srcSpm, SentencePiece? tgtSpm,
            Seq2SeqSharp.Utils.DecodingStrategyEnums decodingStrategyEnum, float topPSampling, float repeatPenalty, float memoryUsageRatio, string mklInstructions, int beamSearchSize, string blockedTokens, ModelType modelType,
            string wordMappingFilePath)
        {
            opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxSrcSentLength = maxTestSrcSentLength;
            opts.MaxTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;
            opts.DecodingStrategy = decodingStrategyEnum;
            opts.DecodingRepeatPenalty = repeatPenalty;
            opts.MemoryUsageRatio = memoryUsageRatio;
            opts.MKLInstructions = mklInstructions;
            opts.BeamSearchSize = beamSearchSize;

            m_srcSpm = srcSpm;
            m_tgtSpm = tgtSpm;
            m_modelType = modelType;

            if (String.IsNullOrEmpty(wordMappingFilePath) == false)
            {
                Logger.WriteLine($"Loading word mapping file from '{wordMappingFilePath}'");
                m_wordMappings = new Dictionary<string, string>();
                foreach (var line in File.ReadLines(wordMappingFilePath))
                {
                    string[] items = line.Split('\t');

                    if (m_wordMappings.ContainsKey(items[1]) == false)
                    {
                        m_wordMappings.Add(items[1], items[0]);
                    }
                    else
                    {
                        Logger.WriteLine(Logger.Level.warn, $"Word '{items[1]}' has been mapped to '{m_wordMappings[items[1]]}', so it cannot be remapped to '{items[0]}'");
                    }
                }

            }


            if (opts.ProcessorType == ProcessorTypeEnums.CPU)
            {
                sm = new Semaphore(Environment.ProcessorCount, Environment.ProcessorCount);
            }
            else
            {
                sm = new Semaphore(1, 1);
            }

            Logger.WriteLine($"Creating Seq2Seq instance. ModelType = '{m_modelType}'");
            if (m_modelType == ModelType.EncoderDecoder)
            {
                m_seq2seq = new Seq2Seq(opts);
            }
            else
            {
                m_seq2seq = new GPT(opts);
            }
            if (String.IsNullOrEmpty(blockedTokens) == false)
            {
                Logger.WriteLine($"Creating blocked tokens = '{blockedTokens}'");
                string[] tokens = blockedTokens.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                
                m_blockedTokens = new List<int>();
                foreach (var token in tokens)
                {
                    m_blockedTokens.Add(int.Parse(token));
                }
            }

        }

        static (bool, string) CheckRepeatSentence(string sent)
        {
            for (int i = 10; i <= sent.Length / 2; i++)
            {
                string tailPart = sent.Substring(sent.Length - i);
                string midPart = sent.Substring(sent.Length - i - tailPart.Length, tailPart.Length);

                if (tailPart == midPart)
                {
                    sent = sent.Substring(0, sent.Length - tailPart.Length);
                    return (true, sent);
                }
            }

            return (false, sent);
        }

        static public string Call(string rawSrcInput, string rawTgtInput, int tokenNumToGenerate, bool random, float repeatPenalty)
        {
            if (opts == null)
            {
                throw new ArgumentNullException($"The {nameof(Seq2SeqInstance)} may not be initialized, and option instance is null.");
            }

            if (m_seq2seq == null)
            {
                throw new ArgumentNullException($"The {nameof(Seq2SeqInstance)} is null.");
            }

            if (m_wordMappings != null)
            {
                lock (locker)
                {
                    foreach (var pair in m_wordMappings)
                    {
                        rawSrcInput = rawSrcInput.Replace(pair.Key, pair.Value);
                    }
                }
            }


            if (m_modelType == ModelType.DecoderOnly && String.IsNullOrEmpty(rawTgtInput))
            {
                rawTgtInput = rawSrcInput;
            }

            var srcInput = (m_srcSpm != null) ? m_srcSpm.Encode(rawSrcInput) : rawSrcInput;
            List<string> srcTokens = srcInput.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
            if (srcTokens.Count > opts.MaxSrcSentLength)
            {
                srcTokens = srcTokens.GetRange(0, opts.MaxSrcSentLength);
            }


            List<List<String>> batchTokens = new List<List<string>>();
            batchTokens.Add(srcTokens);

            List<List<List<string>>> srcGroupBatchTokens = new List<List<List<string>>>();
            srcGroupBatchTokens.Add(batchTokens);


            var tgtInput = (m_tgtSpm != null) ? m_tgtSpm.Encode(rawTgtInput) : rawTgtInput;
            List<string> tgtTokens = tgtInput.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
            if (tgtTokens.Count > opts.MaxTgtSentLength)
            {
                tgtTokens = tgtTokens.GetRange(tgtTokens.Count - opts.MaxTgtSentLength, opts.MaxTgtSentLength);
            }

            tokenNumToGenerate += tgtTokens.Count;

            if (tokenNumToGenerate > opts.MaxTgtSentLength)
            {
                //The target text is too long, so we won't generate any more text for it.
                Logger.WriteLine($"Given target text '{rawTgtInput}' is too long, so we won't generate any more text for it.");
                return rawTgtInput + " EOS";
            }

            List<List<String>> batchTokens2 = new List<List<string>>();
            batchTokens2.Add(tgtTokens);

            List<List<List<string>>> tgtGroupBatchTokens = new List<List<List<string>>>();
            tgtGroupBatchTokens.Add(batchTokens2);


            DecodingOptions decodingOptions = opts.CreateDecodingOptions();
            decodingOptions.MaxTgtSentLength = tokenNumToGenerate;
            decodingOptions.RepeatPenalty = repeatPenalty;
            decodingOptions.RandomSelectOutputToken = random;
            decodingOptions.BlockedTokens = m_blockedTokens;

            if (repeatPenalty == 0)
            {
                decodingOptions.DecodingStrategy = Seq2SeqSharp.Utils.DecodingStrategyEnums.GreedySearch;
            }

            try
            {
                sm?.WaitOne();

                List<NetworkResult> nrs = null;

                if (m_modelType == ModelType.EncoderDecoder)
                {
                    nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>(srcGroupBatchTokens, tgtGroupBatchTokens, decodingOptions);
                }
                else
                {
                    nrs = m_seq2seq.Test<SeqCorpusBatch>(tgtGroupBatchTokens, tgtGroupBatchTokens, decodingOptions);
                }

                string rst = String.Join(" ", nrs[0].Output[0][0].ToArray(), 0, nrs[0].Output[0][0].Count);
                bool isEnded = (rst.EndsWith("</s>") || rst == tgtInput || nrs[0].Status == NetworkResultStatus.OOM);

                rst = (m_tgtSpm != null) ? m_tgtSpm.Decode(rst) : rst;
                (bool isRepeat, rst) = CheckRepeatSentence(rst);

                if (isRepeat)
                {
                    rst = rst + " !!! Found repeat sentences, try to use larger value of penalty for repeat. !!!";
                    isEnded = true;
                }

                if (isEnded)
                {
                    rst += " EOS";
                    Logger.WriteLine($"Completed text generation: Source Input Text = '{rawSrcInput}', IsRandomSample = '{random}', Repeat Penalty = '{repeatPenalty}', Output Text = '{rst}'");
                }

                return rst;
            }
            catch (Exception ex)
            {
                Logger.WriteLine(Logger.Level.err, $"Error Message = '{ex.Message}', Call stack = '{ex.StackTrace}'");

                return rawTgtInput + " EOS";
            }
            finally
            {
                sm?.Release();
            }
        }
    }
}
