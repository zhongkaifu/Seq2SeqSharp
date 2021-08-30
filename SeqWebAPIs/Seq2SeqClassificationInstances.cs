using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SeqWebAPIs
{
    static public class Seq2SeqClassificationInstances
    {
        private static object locker = new object();
        private static Dictionary<string, Seq2SeqClassification> m_key2Instance = new Dictionary<string, Seq2SeqClassification>();

        static public void Initialization(Dictionary<string, string> key2ModelFilePath, int maxTestSrcSentLength, int maxTestTgtSentLength, string processorType, string deviceIds)
        {
            foreach (var pair in key2ModelFilePath)
            {
                Logger.WriteLine($"Loading '{pair.Key}' model from '{pair.Value}'");
                Seq2SeqClassificationOptions opts = new Seq2SeqClassificationOptions();
                opts.ModelFilePath = pair.Value;
                opts.MaxTestSrcSentLength = maxTestSrcSentLength;
                opts.MaxTestTgtSentLength = maxTestTgtSentLength;
                opts.ProcessorType = processorType;
                opts.DeviceIds = deviceIds;

                var inst = new Seq2SeqClassification(opts);

                m_key2Instance.Add(pair.Key, inst);
            }
        }
        static public (string, string) Call(string key, List<string> inputFeatureGroups)
        {
            if (m_key2Instance.ContainsKey(key) == false)
            {
                return ("", "");
            }

            List<List<List<string>>> groupBatchTokens = new List<List<List<string>>>();
            foreach (var inputFeatureGroup in inputFeatureGroups)
            {
                List<string> tokens = inputFeatureGroup.Split(' ').ToList();
                List<List<string>> batchTokens = new List<List<string>>();
                batchTokens.Add(tokens);
                groupBatchTokens.Add(batchTokens);
            }

            lock (locker)
            {
                List<NetworkResult> nrs = m_key2Instance[key].Test(groupBatchTokens);
                var nrCLS = nrs[0];
                var nrSeq2Seq = nrs[1];

                string tag = nrCLS.Output[0][0][0];
                string text = String.Join(" ", nrSeq2Seq.Output[0][0].ToArray(), 1, nrSeq2Seq.Output[0][0].Count - 2);

                return (tag, text);
            }
        }
    }
}
