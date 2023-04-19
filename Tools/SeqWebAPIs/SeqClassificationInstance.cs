using System.Collections.Generic;
using System.Linq;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;

namespace SeqClassificationWebAPI
{
    public static class SeqClassificationInstance
    {
        static private object locker = new object();
        static private SeqClassification m_seqClassification;
        static private SeqClassificationOptions opts;

        static public void Initialization(string modelFilePath, int maxTestSentLength, ProcessorTypeEnums processorType, string deviceIds)
        {
            opts = new SeqClassificationOptions();
            opts.ModelFilePath = modelFilePath;
            opts.MaxSentLength = maxTestSentLength;
            opts.ProcessorType = processorType;
            opts.DeviceIds = deviceIds;

            m_seqClassification = new SeqClassification(opts);
        }

        static public string Call(List<string> inputFeatureGroups)
        {
            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();
            foreach (var inputFeatureGroup in inputFeatureGroups)
            {
                List<string> tokens = inputFeatureGroup.Split(' ').ToList();
                List<List<string>> batchTokens = new List<List<string>>();
                batchTokens.Add(tokens);
                groupBatchTokens.Add(batchTokens);
            }

            DecodingOptions decodingOptions = opts.CreateDecodingOptions();
            List<NetworkResult> nrs = m_seqClassification.Test<SeqClassificationMultiTasksCorpusBatch>(groupBatchTokens, null, decodingOptions);

            List<string> tags = new List<string>();
            foreach (var nr in nrs)
            {
                tags.Add(nr.Output[0][0][0]); // shape: (beam_size, batch_size, seq_size)
            }

            return string.Join("\t", tags);
        }
    }
}
