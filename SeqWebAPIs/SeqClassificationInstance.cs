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
        static public void Initialization(string modelFilePath, int maxTestSentLength, string processorType, string deviceIds)
        {
            SeqClassificationOptions opts = new SeqClassificationOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSentLength = maxTestSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            m_seqClassification = new SeqClassification(opts);
        }

        static public string Call(string input1, string input2)
        {
            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();

            // Build features in group 1
            List<string> tokens = input1.Split(' ').ToList();
            List<List<String>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);
            groupBatchTokens.Add(batchTokens);

            // Build features in group 2
            tokens = input2.Split(' ').ToList();
            batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);
            groupBatchTokens.Add(batchTokens);


            lock (locker)
            {
                List<NetworkResult> nrs = m_seqClassification.Test(groupBatchTokens);

                List<string> tags = new List<string>();
                foreach (var nr in nrs)
                {
                    tags.Add(nr.Output[0][0][0]); // shape: (beam_size, batch_size, seq_size)
                }

                return String.Join("\t", tags);
            }
        }
    }
}
