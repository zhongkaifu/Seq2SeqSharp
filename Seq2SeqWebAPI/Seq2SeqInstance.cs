using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Seq2SeqSharp;
using Seq2SeqSharp.Tools;
using AdvUtils;

namespace Seq2SeqWebAPI
{
    public static class Seq2SeqInstance
    {
        static private object locker = new object();
        static private Seq2Seq m_seq2seq;
        static public void Initialization(string modelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, string processorType)
        {
            Seq2SeqOptions opts = new Seq2SeqOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSrcSentLength = maxTestSrcSentLength;
            opts.MaxTestTgtSentLength = maxTestTgtSentLength;
            opts.ProcessorType = processorType;

            m_seq2seq = new Seq2Seq(opts);
        }

        static public string Call(string input)
        {
            List<string> tokens = input.Split(' ').ToList();
            List<List<String>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);

            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();
            groupBatchTokens.Add(batchTokens);

            lock (locker)
            {
                NetworkResult nr = m_seq2seq.Test(groupBatchTokens);
                string rst = String.Join(" ", nr.Output[0][0].ToArray(), 1, nr.Output[0][0].Count - 2);

                return rst;
            }
        }
    }
}
