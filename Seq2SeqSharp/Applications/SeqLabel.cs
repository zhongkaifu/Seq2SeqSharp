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
using System.IO;
using System.Linq;

using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using TensorSharp;

namespace Seq2SeqSharp
{
    public class SeqLabel : BaseSeq2SeqFramework<SeqLabelModel>
    {
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_ffLayer; //The feed forward layers over over devices.
        private MultiProcessorNetworkWrapper<IWeightTensor> m_segmentEmbedding;

        private MultiProcessorNetworkWrapper<IWeightTensor> m_posEmbedding = null;

        private readonly PaddingEnums m_paddingType = PaddingEnums.AllowPadding;
        private readonly SeqLabelOptions m_options;

        private readonly float[] m_tagWeightsList = null;

        public SeqLabel(SeqLabelOptions options, Vocab srcVocab = null, Vocab clsVocab = null)
            : base(options.DeviceIds, options.ProcessorType, options.ModelFilePath, options.MemoryUsageRatio, options.CompilerOptions, startToRunValidAfterUpdates: options.StartValidAfterUpdates,
                  runValidEveryUpdates: options.RunValidEveryUpdates, updateFreq: options.UpdateFreq, maxDegressOfParallelism: options.TaskParallelism, 
                  cudaMemoryAllocatorType: options.CudaMemoryAllocatorType, elementType: options.AMP ? DType.Float16 : DType.Float32, saveModelEveryUpdats: options.SaveModelEveryUpdates, 
                  initLossScaling: options.InitLossScaling, autoCheckTensorCorruption: options.CheckTensorCorrupted)
        {
            m_paddingType = options.PaddingType;
            m_options = options;

            // Check if options are valided.
            m_options.ValidateOptions();

            if (File.Exists(m_options.ModelFilePath))
            {
                if (srcVocab != null || clsVocab != null)
                {
                    throw new ArgumentException($"Model '{m_options.ModelFilePath}' exists and it includes vocabulary, so input vocabulary must be null.");
                }

                // Model file exists, so we load it from file.
                m_modelMetaData = LoadModel();
            }
            else
            {
                // Model doesn't exist, we create it and initlaize parameters
                m_modelMetaData = new SeqLabelModel(options, srcVocab, clsVocab);

                //Initializng weights in encoders and decoders
                CreateTrainableParameters(m_modelMetaData);
            }

            m_modelMetaData.ShowModelInfo();

            if (String.IsNullOrEmpty(m_options.TagWeights) == false)
            {
                m_tagWeightsList = SplitTagWeights(m_options.TagWeights);

                //TODO(Zho): the for loop is executed even if nothing is printed
                {
                    Logger.WriteLine(Logger.Level.debug, "The list of tag weights:");
                    for (int i = 0; i < m_tagWeightsList.Length; i++)
                    {
                        Logger.WriteLine(Logger.Level.debug, $"{i}:{m_tagWeightsList[i]}");
                    }
                }
            }
            else
            {
                Logger.WriteLine(Logger.Level.debug, "No tag weights are specified.");

                m_tagWeightsList = null;
            }
        }

        protected override SeqLabelModel LoadModel(string suffix = "") => base.LoadModelRoutine<Model_4_ProtoBufSerializer>(CreateTrainableParameters, SeqLabelModel.Create, suffix);
        private bool CreateTrainableParameters(IModel model)
        {
            Logger.WriteLine(Logger.Level.debug, $"Creating encoders and decoders...");

            var raDeviceIds = new RoundArray<int>(DeviceIds);
            m_encoder = Encoder.CreateEncoders(model, m_options, raDeviceIds);
            m_ffLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", model.HiddenDim, model.TgtVocab.Count, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem(), isTrainable: true), DeviceIds);

            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { model.SrcVocab.Count, model.EncoderEmbeddingDim }, raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, name: "SrcEmbeddings", 
                isTrainable: true), DeviceIds);
            (m_posEmbedding, m_segmentEmbedding) = Misc.CreateAuxEmbeddings(raDeviceIds, model.HiddenDim, m_options.MaxSentLength, model, createAPE: (model.PEType == PositionEmbeddingEnums.APE));

            return true;
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        private (IEncoder, IWeightTensor, IWeightTensor, FeedForwardLayer, IWeightTensor) GetNetworksOnDeviceAt(int deviceId)
        {
            var deviceIdIdx = TensorAllocator.GetDeviceIdIndex(deviceId);

            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx),
                m_segmentEmbedding?.GetNetworkOnDevice(deviceIdIdx), m_ffLayer.GetNetworkOnDevice(deviceIdIdx), m_posEmbedding?.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="g">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side. In training mode, it inputs target tokens, otherwise, it outputs target tokens generated by decoder</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        public override List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph g, IPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
        {
            List<NetworkResult> nrs = new List<NetworkResult>();

            var srcSnts = sntPairBatch.GetSrcTokens();
            var tgtSnts = sntPairBatch.GetTgtTokens();

            (var encoder, var srcEmbedding, var segmentEmbedding, var decoderFFLayer, var posEmbeddings) = GetNetworksOnDeviceAt(g.DeviceId);

            // Reset networks
            encoder.Reset(g.GetWeightFactory(), srcSnts.Count);

            var originalSrcLengths = BuildInTokens.PadSentences(srcSnts);
            var srcTokensList = m_modelMetaData.SrcVocab.GetWordIndex(srcSnts);


            //if (isTraining)
            //{
            //    if (srcTokensList.Count != tgtTokensList.Count)
            //    {
            //        throw new InvalidDataException($"Inconsistent batch size between source and target. source batch size = '{srcTokensList.Count}', target batch size = '{tgtTokensList.Count}'");
            //    }

            //    for (int i = 0; i < srcTokensList.Count; i++)
            //    {
            //        if (srcTokensList[i].Count != tgtTokensList[i].Count)
            //        {
            //            var srcWords = m_modelMetaData.SrcVocab.ConvertIdsToString(srcTokensList[i]);
            //            var tgtWords = m_modelMetaData.ClsVocab.ConvertIdsToString(tgtTokensList[i]);

            //            throw new InvalidDataException($"Inconsistent sequence length between source and target at batch '{i}'. source sequence length = '{srcTokensList[i].Count}', target sequence length = '{tgtTokensList[i].Count}' src sequence = '{String.Join(" ", srcWords)}', tgt sequence = '{String.Join(" ", tgtWords)}'");
            //        }
            //    }
            //}

            int seqLen = srcSnts[0].Count;
            int batchSize = srcSnts.Count;

            // Encoding input source sentences
            IWeightTensor encOutput = Encoder.Run(g, encoder, m_modelMetaData, m_paddingType, srcEmbedding, posEmbeddings, segmentEmbedding, srcTokensList, originalSrcLengths);
            IWeightTensor ffLayer = decoderFFLayer.Process(encOutput, batchSize, g);

            float cost = 0.0f;
            IWeightTensor probs = g.Softmax(ffLayer, inPlace: true);

            if (isTraining)
            {
                BuildInTokens.PadSentences(tgtSnts);
                var tgtTokensList = m_modelMetaData.TgtVocab.GetWordIndex(tgtSnts);
                var tgtTokensIdxTensor = g.CreateTensorForIndex(tgtTokensList);

                if (m_tagWeightsList == null)
                {
                    cost = g.CrossEntropyLoss(probs, tgtTokensIdxTensor, label_smoothing: m_options.LabelSmoothing, graident: LossScaling);
                }
                else
                {
                    var tagWeightsTensor = g.CreateTensorWeights(sizes: new long[] { 1, m_tagWeightsList.Length }, m_tagWeightsList, dtype: probs.ElementType);
                    tagWeightsTensor = g.Expand(tagWeightsTensor, dims: probs.Sizes);
                    cost = g.CrossEntropyLoss(probs, tgtTokensIdxTensor, tagWeightsTensor, label_smoothing: m_options.LabelSmoothing);
                }
            }
            else
            {
                // Output "i"th target word
                using var targetIdxTensor = g.Argmax(probs, 1);
                float[] targetIdx = targetIdxTensor.ToWeightArray();
                List<string> targetWords = m_modelMetaData.TgtVocab.ConvertIdsToString(targetIdx.ToList());

                for (int k = 0; k < batchSize; k++)
                {
                    tgtSnts[k] = targetWords.GetRange(k * seqLen, seqLen);
                }
            }

            NetworkResult nr = new NetworkResult
            {
                Cost = cost,
                Output = new List<List<List<string>>>()
            };
            nr.Output.Add(tgtSnts);

            nrs.Add(nr);

            return nrs;
        }


        public float[] SplitTagWeights(string tagWeights)
        {
            Vocab clsVocab = m_modelMetaData.TgtVocab;
            float[] array = new float[clsVocab.Count];
            Array.Clear(array, 0, array.Length);

            string[] tagWeightArray = tagWeights.Split(',');
            foreach(var tagWeight in tagWeightArray)
            {
                string[] pair = tagWeight.Trim().Split(':');
                int idx = clsVocab.GetWordIndex(pair[0]);
                array[idx] = float.Parse(pair[1]);

            }

            return array;
        }

        public void DumpVocabToFiles(string outputSrcVocab, string outputTgtVocab)
        {
            m_modelMetaData.SrcVocab.DumpVocab(outputSrcVocab);
            m_modelMetaData.TgtVocab.DumpVocab(outputTgtVocab);
        }
    }
}
