using System;
using System.Collections.Generic;
using System.Linq;

using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;

namespace Seq2SeqSharp.Utils
{
    public class RoundArray<T>
    {
        private readonly T[] m_array;
        private int currentIdx = 0;
        public RoundArray(T[] a)
        {
            m_array = a;
        }

        public T GetNextItem()
        {
            T item = m_array[currentIdx];
            currentIdx = (currentIdx + 1) % m_array.Length;

            return item;
        }

        public T[] ToArray()
        {
            return m_array;
        }
    }

    public static class Utils
    {
        public static string GetTimeStamp(DateTime timeStamp)
        {
            return string.Format("{0:yyyy}_{0:MM}_{0:dd}_{0:HH}h_{0:mm}m_{0:ss}s", timeStamp);
        }
    }

    public static class Misc
    {
        public static void AppendNewBatch(List<List<List<string>>> inputBatchs, string line, int maxTokenLength)
        {
            string[] groups = line.Trim().Split('\t');

            if (inputBatchs.Count == 0)
            {
                for (int i = 0; i < groups.Length; i++)
                {
                    inputBatchs.Add(new List<List<string>>());
                }
            }

            for (int i = 0; i < groups.Length; i++)
            {
                var group = groups[i];
                List<string> tokens = group.Trim().Split(' ').ToList();
                if (tokens.Count > maxTokenLength - 2)
                {
                    tokens = tokens.GetRange(0, maxTokenLength - 2);
                }
                inputBatchs[i].Add(tokens);
            }
        }


        public static void Ss_StatusUpdateWatcher(object sender, EventArgs e)
        {
            CostEventArg ep = e as CostEventArg;

            TimeSpan ts = DateTime.Now - ep.StartDateTime;
            double sentPerMin = 0;
            double wordPerSec = 0;
            if (ts.TotalMinutes > 0)
            {
                sentPerMin = ep.ProcessedSentencesInTotal / ts.TotalMinutes;
            }

            if (ts.TotalSeconds > 0)
            {
                wordPerSec = ep.ProcessedWordsInTotal / ts.TotalSeconds;
            }

            Logger.WriteLine($"Update = {ep.Update}, Epoch = {ep.Epoch}, LR = {ep.LearningRate.ToString("e4")}, AvgCost = {ep.AvgCostInTotal.ToString("e4")}, Sent = {ep.ProcessedSentencesInTotal}, SentPerMin = {sentPerMin:F}, WordPerSec = {wordPerSec:F}");
        }

        public static IOptimizer CreateOptimizer(Options opts)
        {
            // Create optimizer
            IOptimizer optimizer = null;
            if (string.Equals(opts.Optimizer, "Adam", StringComparison.InvariantCultureIgnoreCase))
            {
                optimizer = new AdamOptimizer(opts.GradClip, opts.Beta1, opts.Beta2);
            }
            else
            {
                optimizer = new RMSPropOptimizer(opts.GradClip, opts.Beta1);
            }

            return optimizer;
        }

        public static (MultiProcessorNetworkWrapper<IWeightTensor>, MultiProcessorNetworkWrapper<IWeightTensor>) CreateAuxEmbeddings(RoundArray<int> raDeviceIds, int hiddenDim, int maxSentLength, IModel modelMetaData)
        {
            MultiProcessorNetworkWrapper<IWeightTensor> posEmbeddings = null;
            MultiProcessorNetworkWrapper<IWeightTensor> segmentEmbeddings = null;

            if (modelMetaData.EncoderType == EncoderTypeEnums.Transformer || modelMetaData.DecoderType == DecoderTypeEnums.Transformer)
            {
                posEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                    maxSentLength + 2,
                    hiddenDim, raDeviceIds.GetNextItem(), "PosEmbedding", false), raDeviceIds.ToArray(), true);

                if (modelMetaData.EnableSegmentEmbeddings)
                {
                    segmentEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.MaxSegmentNum, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), normType: NormType.Uniform, name: "SegmentEmbedding", isTrainable: true), raDeviceIds.ToArray());
                }
            }

            return (posEmbeddings, segmentEmbeddings);
        }


        [M(O.AggressiveInlining)] public static bool IsNullOrEmpty( this string s ) => string.IsNullOrEmpty( s );
        [M(O.AggressiveInlining)] public static bool IsNullOrWhiteSpace( this string s ) => string.IsNullOrWhiteSpace( s );
    }
}
