using System;
using System.Collections.Generic;
using System.Linq;

using Seq2SeqSharp;
using Seq2SeqSharp.Corpus;

namespace Seq2SeqWebAPI
{
    public static class Seq2SeqInstance
    {
        static private object locker = new object();
        static private Seq2Seq m_seq2seq;
        static public void Initialization(string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, ProcessorTypeEnums processorType, string deviceIds)
        {
            var opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSrcSentLength = maxTestSrcSentLength;
            opts.MaxTestTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            m_seq2seq = new Seq2Seq(opts);
        }

        static public string Call(string input)
        {
            List<string> tokens = input.Split(' ').ToList();
            List<List<string>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);

            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();
            groupBatchTokens.Add(batchTokens);

            lock (locker)
            {
                var nrs = m_seq2seq.Test<Seq2SeqCorpusBatch>(groupBatchTokens);
                string rst = string.Join(" ", nrs[0].Output[0][0].ToArray(), 1, nrs[0].Output[0][0].Count - 2);

                return rst;
            }
        }
    }
}
