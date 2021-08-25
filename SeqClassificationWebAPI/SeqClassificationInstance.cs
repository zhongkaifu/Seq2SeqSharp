using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeqClassificationWebAPI
{
    public static class SeqClassificationInstance
    {
        static private object locker = new object();
        static private SeqClassification m_seqClassification;
        static public void Initialization(string modelFilePath, int maxTestSentLength, string processorType)
        {
            SeqClassificationOptions opts = new SeqClassificationOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSentLength = maxTestSentLength;
            opts.ProcessorType = processorType;

            m_seqClassification = new SeqClassification(opts);
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
                List<NetworkResult> nrs = m_seqClassification.Test(groupBatchTokens);

                List<string> tags = new List<string>();
                foreach (var nr in nrs)
                {
                    tags.Add(nr.Output[0][0][0]);
                }

                return String.Join("\t", tags);
            }
        }
    }
}
