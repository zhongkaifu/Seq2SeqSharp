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

namespace Seq2SeqWebApps
{
    public static class Seq2SeqInstance
    {
        static private Seq2Seq? m_seq2seq;
        static private SentencePiece? m_srcSpm = null;
        static private SentencePiece? m_tgtSpm = null;
        static private Seq2SeqOptions? opts;
        static Semaphore? sm = null;

        static public void Initialization(string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, ProcessorTypeEnums processorType, string deviceIds, SentencePiece? srcSpm, SentencePiece? tgtSpm,
            Seq2SeqSharp.Utils.DecodingStrategyEnums decodingStrategyEnum, float topPSampling, float repeatPenalty, float memoryUsageRatio, string mklInstructions)
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

            m_srcSpm = srcSpm;
            m_tgtSpm = tgtSpm;

            if (opts.ProcessorType == ProcessorTypeEnums.CPU)
            {
                sm = new Semaphore(Environment.ProcessorCount, Environment.ProcessorCount);
            }
            else
            {
                sm = new Semaphore(1, 1);
            }

            m_seq2seq = new Seq2Seq(opts);
        }

        static (bool, string) CheckRepeatSentence(string sent)
        {
            for (int i = 5; i <= sent.Length / 2; i++)
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

            var srcInput = (m_srcSpm != null) ? m_srcSpm.Encode(rawSrcInput) : rawSrcInput;
            List<string> tokens = srcInput.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();

            if (tokens.Count > opts.MaxSrcSentLength)
            {
                tokens = tokens.GetRange(tokens.Count - opts.MaxSrcSentLength, opts.MaxSrcSentLength);
            }


            List<List<String>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);

            List<List<List<string>>> srcGroupBatchTokens = new List<List<List<string>>>();
            srcGroupBatchTokens.Add(batchTokens);


            var tgtInput = (m_tgtSpm != null) ? m_tgtSpm.Encode(rawTgtInput) : rawTgtInput;
            List<string> tokens2 = tgtInput.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
            tokenNumToGenerate += tokens2.Count;

            if (tokenNumToGenerate > opts.MaxTgtSentLength)
            {
                //The target text is too long, so we won't generate any more text for it.
                Logger.WriteLine($"Given target text '{rawTgtInput}' is too long, so we won't generate any more text for it.");
                return rawTgtInput + " EOS";
            }

            List<List<String>> batchTokens2 = new List<List<string>>();
            batchTokens2.Add(tokens2);

            List<List<List<string>>> tgtGroupBatchTokens = new List<List<List<string>>>();
            tgtGroupBatchTokens.Add(batchTokens2);


            DecodingOptions decodingOptions = opts.CreateDecodingOptions();
            decodingOptions.MaxTgtSentLength = tokenNumToGenerate;
            decodingOptions.RepeatPenalty = repeatPenalty;

            try
            {
                sm?.WaitOne();

                var nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>(srcGroupBatchTokens, tgtGroupBatchTokens, decodingOptions);
                string rst = String.Join(" ", nrs[0].Output[0][0].ToArray(), 0, nrs[0].Output[0][0].Count);
                bool isEnded = (rst.EndsWith("</s>") || rst == tgtInput);
                rst = (m_tgtSpm != null) ? m_tgtSpm.Decode(rst) : rst;
                (bool isRepeat, string truncatedStr) = CheckRepeatSentence(rst);

                if (isRepeat)
                {
                    rst = truncatedStr + " REPEAT";
                    isEnded = true;
                }

                if (isEnded)
                {
                    rst += " EOS";
                    Logger.WriteLine($"Completed text generation: Source Input Text = '{rawSrcInput}', Target Prompt Text = '{rawTgtInput}', Token Numbers To Generate = '{tokenNumToGenerate}', IsRandomSample = '{random}', Repeat Penalty = '{repeatPenalty}', Output Text = '{rst}'");
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
