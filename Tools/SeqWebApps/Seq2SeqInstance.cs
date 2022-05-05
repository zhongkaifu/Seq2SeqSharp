using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Seq2SeqSharp;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Applications;

namespace Seq2SeqWebApps
{
    public static class Seq2SeqInstance
    {
        static private Seq2Seq? m_seq2seq;
        static private SentencePiece? m_srcSpm = null;
        static private SentencePiece? m_tgtSpm = null;
        static private Seq2SeqOptions? opts;


        static public void Initialization(string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, ProcessorTypeEnums processorType, string deviceIds, SentencePiece? srcSpm, SentencePiece? tgtSpm,
            Seq2SeqSharp.Utils.DecodingStrategyEnums decodingStrategyEnum, float topPSampling, float repeatPenalty)
        {
            opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSrcSentLength = maxTestSrcSentLength;
            opts.MaxTestTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;
            opts.DecodingStrategy = decodingStrategyEnum;
            opts.DecodingRepeatPenalty = repeatPenalty;
            opts.DecodingTopPValue = topPSampling;

            m_srcSpm = srcSpm;
            m_tgtSpm = tgtSpm;

            m_seq2seq = new Seq2Seq(opts);
        }

        static public string Call(string srcInput, string tgtInput, int tokenNumToGenerate, bool random, float repeatPenalty)
        {
            if (opts == null)
            {
                throw new ArgumentNullException($"The {nameof(Seq2SeqInstance)} may not be initialized, and option instance is null.");
            }

            if (m_seq2seq == null)
            {
                throw new ArgumentNullException($"The {nameof(Seq2SeqInstance)} is null.");
            }

            srcInput = (m_srcSpm != null) ? m_srcSpm.Encode(srcInput) : srcInput;
            List<string> tokens = srcInput.Split(' ').ToList();

            if (tokens.Count > opts.MaxTestSrcSentLength)
            {
                tokens = tokens.GetRange(tokens.Count - opts.MaxTestSrcSentLength, opts.MaxTestSrcSentLength);
            }


            List<List<String>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);

            List<List<List<string>>> srcGroupBatchTokens = new List<List<List<string>>>();
            srcGroupBatchTokens.Add(batchTokens);


            tgtInput = (m_tgtSpm != null) ? m_tgtSpm.Encode(tgtInput) : tgtInput;
            List<string> tokens2 = tgtInput.Split(' ').ToList();
            tokenNumToGenerate += tokens2.Count;

            List<List<String>> batchTokens2 = new List<List<string>>();
            batchTokens2.Add(tokens2);

            List<List<List<string>>> tgtGroupBatchTokens = new List<List<List<string>>>();
            tgtGroupBatchTokens.Add(batchTokens2);


            DecodingOptions decodingOptions = opts.CreateDecodingOptions();
            decodingOptions.MaxTgtSentLength = tokenNumToGenerate;
            decodingOptions.TopPValue = random ? 0.5f : 0.0f;
            decodingOptions.RepeatPenalty = repeatPenalty;

            var nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>(srcGroupBatchTokens, tgtGroupBatchTokens, decodingOptions);
            string rst = String.Join(" ", nrs[0].Output[0][0].ToArray(), 0, nrs[0].Output[0][0].Count);
            rst = (m_tgtSpm != null) ? m_tgtSpm.Decode(rst) : rst;

            return rst;
        }
    }
}
