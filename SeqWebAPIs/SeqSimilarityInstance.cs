using System;
using System.Collections.Generic;
using System.Linq;

using Seq2SeqSharp;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Tools;

namespace SeqSimilarityWebAPI
{
    static public class SeqSimilarityInstance
    {
        static private object locker = new object();
        static private SeqSimilarity m_seqSimilarity;
        static public void Initialization(string modelFilePath, int maxTestSentLength, ProcessorTypeEnums processorType, string deviceIds)
        {
            SeqSimilarityOptions opts = new SeqSimilarityOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxTestSentLength = maxTestSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            m_seqSimilarity = new SeqSimilarity(opts);
        }

        static public string Call(string input1, string input2)
        {
            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();

            // Build group 1 features for input string 1
            List<string> tokens = input1.Split(' ').ToList();
            List<List<string>> batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);
            groupBatchTokens.Add(batchTokens);

            // Build group 2 features for input string 2
            tokens = input2.Split(' ').ToList();
            batchTokens = new List<List<string>>();
            batchTokens.Add(tokens);
            groupBatchTokens.Add(batchTokens);


            lock (locker)
            {
                List<NetworkResult> nrs = m_seqSimilarity.Test<SeqClassificationMultiTasksCorpusBatch>(groupBatchTokens);

                List<string> tags = new List<string>();
                foreach (var nr in nrs)
                {
                    tags.Add(nr.Output[0][0][0]); // shape: (beam_size, batch_size, seq_size)
                }

                return string.Join("\t", tags);
            }
        }
    }
}
