// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;
using Seq2SeqSharp.Enums;
using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Tools;
using TensorSharp;
using M = System.Runtime.CompilerServices.MethodImplAttribute;
using O = System.Runtime.CompilerServices.MethodImplOptions;
using ManagedCuda;

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

        /// <summary>
        /// Get the number of GPU's or CPU cores in the system
        /// </summary>
        /// <param name="GPU">true: get the number of GPUs in the system (default), false: get the number of CPU cores in the system</param>
        /// <returns>number of GPUs or CPU cores in the system</returns>
        public static int GetDeviceCount(bool GPU = true)
        {
            try
            {
                if (GPU)
                {
                    return CudaContext.GetDeviceCount();
                }
                else
                {
                    return Environment.ProcessorCount;
                }
            }
            catch (Exception)
            {
                return 0;
            }
        }
    }

    public static class Misc
    {
        public static void AppendNewBatch(List<List<string>> inputBatchs, string line, int maxTokenLength)
        {
            List<string> tokens = line.Trim().Split(' ').ToList();
            if (tokens.Count > maxTokenLength - 2)
            {
                tokens = tokens.GetRange(0, maxTokenLength - 2);
            }
            inputBatchs.Add(tokens);            
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
                optimizer = new AdamOptimizer(opts.GradClip, opts.Beta1, opts.Beta2, opts.SaveGPUMemoryMode, opts.CheckTensorCorrupted);
            }
            else
            {
                optimizer = new RMSPropOptimizer(opts.GradClip, opts.Beta1);
            }

            return optimizer;
        }

        public static (MultiProcessorNetworkWrapper<IWeightTensor>, MultiProcessorNetworkWrapper<IWeightTensor>) CreateAuxEmbeddings(RoundArray<int> raDeviceIds, int hiddenDim, int maxSentLength, IModel modelMetaData, DType elementType = DType.Float32, bool isTrainable = true, bool createAPE = false)
        {
            MultiProcessorNetworkWrapper<IWeightTensor> posEmbeddings = null;
            MultiProcessorNetworkWrapper<IWeightTensor> segmentEmbeddings = null;

            if (modelMetaData.EncoderType != EncoderTypeEnums.BiLSTM || modelMetaData.DecoderType != DecoderTypeEnums.AttentionLSTM)
            {
                if (createAPE)
                {
                    posEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(PositionEmbedding.BuildPositionWeightTensor(
                        maxSentLength + 2,
                        hiddenDim, raDeviceIds.GetNextItem(), "PosEmbedding", false, elementType: elementType), raDeviceIds.ToArray(), true);
                }

                if (modelMetaData.EnableSegmentEmbeddings)
                {
                    segmentEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.MaxSegmentNum, modelMetaData.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, name: "SegmentEmbedding",
                        isTrainable: isTrainable, dtype: elementType), raDeviceIds.ToArray());
                }
            }

            return (posEmbeddings, segmentEmbeddings);
        }


        [M(O.AggressiveInlining)] public static bool IsNullOrEmpty( this string s ) => string.IsNullOrEmpty( s );
        [M(O.AggressiveInlining)] public static bool IsNullOrWhiteSpace( this string s ) => string.IsNullOrWhiteSpace( s );
    }
}
