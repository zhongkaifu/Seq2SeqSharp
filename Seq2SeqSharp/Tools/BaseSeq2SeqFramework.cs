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
using System.Net.NetworkInformation;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;
using TensorSharp;
using TensorSharp.CUDA.ContextState;
using static ProtoBuf.Meta.RuntimeTypeModel;

namespace Seq2SeqSharp.Tools
{
    public enum NetworkResultStatus
    {
        SUCCEED,
        FAILED,
        OOM
    }

    public class NetworkResult
    {
        public float Cost;
        public List<List<List<string>>> Output; // (beam_size, batch_size, seq_len)
        public List<List<List<int>>> Alignments; // (beam_size, batch_size, seq_len)
        public List<List<List<float>>> AlignmentScores; // (beam_size, batch_size, seq_len)
        public NetworkResultStatus Status;


        public NetworkResult()
        {
            Output = null;
            Alignments = null;
            AlignmentScores = null;
            Status = NetworkResultStatus.FAILED;
        }

        public void RemoveDuplicatedEOS()
        {
            if (Output != null)
            {
                foreach (var item in Output)
                {
                    RemoveDuplicatedEOS(item);
                }
            }
        }

        private static void RemoveDuplicatedEOS(List<List<string>> snts)
        {
            foreach (var snt in snts)
            {
                for (int i = 0; i < snt.Count; i++)
                {
                    if (snt[i] == BuildInTokens.EOS)
                    {
                        snt.RemoveRange(i, snt.Count - i);
                        snt.Add(BuildInTokens.EOS);
                        break;
                    }
                }
            }
        }

        public void AppendResult(NetworkResult nr)
        {
            Status = nr.Status;
            while (Output.Count < nr.Output.Count)
            {
                Output.Add(new List<List<string>>());
            }

            for (int beamIdx = 0; beamIdx < nr.Output.Count; beamIdx++)
            {
                for (int batchIdx = 0; batchIdx < nr.Output[beamIdx].Count; batchIdx++)
                {
                    Output[beamIdx].Add(nr.Output[beamIdx][batchIdx]);
                }

            }
        }
    }

    /// <summary>
    /// This is a framework for neural network training. It includes many core parts, such as backward propagation, parameters updates, 
    /// memory management, computing graph managment, corpus shuffle & batching, I/O for model, logging & monitoring, checkpoints.
    /// You need to create your network inherited from this class, implmenet forward part only and pass it to TrainOneEpoch method for training
    /// </summary>
    public abstract class BaseSeq2SeqFramework<T> where T : Model
    {
        public event EventHandler StatusUpdateWatcher;
        public event EventHandler EvaluationWatcher;
        public event EventHandler EpochEndWatcher;

        private readonly int[] m_deviceIds;
        internal T m_modelMetaData;

        public int[] DeviceIds => m_deviceIds;
        private string m_modelFilePath;
        private readonly float m_regc = 1e-10f; // L2 regularization strength
        private int m_weightsUpdateCount;
        private double m_avgCostPerWordInTotalInLastEpoch = 10000.0;
        private Dictionary<string, double> m_bestPrimaryScoreDict = new Dictionary<string, double>();
        private readonly int m_primaryTaskId = 0;
        private readonly object locker = new object();
        private SortedList<string, IMultiProcessorNetworkWrapper> m_name2network;
        private int m_updateFreq = 1;
        private int m_startToRunValidAfterUpdates = 20000;
        private int m_runValidEveryUpdates = 10000;
        private int m_saveModelEveryUpdates = 0;
        private int m_maxDegressOfParallelism = 1;
        ProcessorTypeEnums m_processorType;
        float m_memoryUsageRatio = 0.9f;
        string m_compilerOptions = null;
        string m_mklInstructions = "AVX2";
        bool m_enableTensorCore = true;
        bool m_saveGPUMemoryMode = false;
        CudaMemoryDeviceAllocatorType m_cudaMemoryAllocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool;
        DType m_elementType = DType.Float32;
        float m_initLossScaling = 1.0f;
        bool m_autoCheckTensorCorruption = false;

        public float LossScaling = 1.0f;

        public BaseSeq2SeqFramework(string deviceIds, ProcessorTypeEnums processorType, string modelFilePath, float memoryUsageRatio = 0.9f, 
            string compilerOptions = null, int runValidEveryUpdates = 10000, int primaryTaskId = 0, int updateFreq = 1, int startToRunValidAfterUpdates = 0,
            int maxDegressOfParallelism = 1, string mklInstructions = "AVX2", int weightsUpdateCount = 0, bool enableTensorCore = true, CudaMemoryDeviceAllocatorType cudaMemoryAllocatorType = CudaMemoryDeviceAllocatorType.CudaMemoryPool, 
            DType elementType = DType.Float32, int randomSeed = -1, int saveModelEveryUpdats = 10000, bool saveGPUMemoryMode = false, float initLossScaling = 1.0f, bool autoCheckTensorCorruption = false)
        {
            m_deviceIds = deviceIds.Split(',').Select(x => int.Parse(x)).ToArray();
            m_compilerOptions = compilerOptions;
            m_modelFilePath = modelFilePath;
            m_processorType= processorType;
            m_memoryUsageRatio = memoryUsageRatio;
            m_mklInstructions = mklInstructions;
            m_enableTensorCore = enableTensorCore;
            m_cudaMemoryAllocatorType = cudaMemoryAllocatorType;
            m_elementType = elementType;
            m_primaryTaskId = primaryTaskId;
            m_updateFreq = updateFreq;
            m_startToRunValidAfterUpdates = startToRunValidAfterUpdates;
            m_runValidEveryUpdates = runValidEveryUpdates;
            m_maxDegressOfParallelism = maxDegressOfParallelism;
            m_weightsUpdateCount = weightsUpdateCount;
            m_saveModelEveryUpdates = saveModelEveryUpdats;
            m_saveGPUMemoryMode = saveGPUMemoryMode;
            m_initLossScaling = initLossScaling;
            m_autoCheckTensorCorruption = autoCheckTensorCorruption;

            InitDevices();

            if (randomSeed == -1)
            {
                randomSeed = DateTime.Now.Millisecond;
            }
            RandomGenerator.Init(randomSeed);
        }

        public void InitDevices()
        {
            string[] cudaCompilerOptions = m_compilerOptions.IsNullOrEmpty() ? null : Regex.Split(m_compilerOptions, "--").ToList().Where(item => item != "").Select(item => "--" + item).ToArray();
            TensorAllocator.InitDevices(m_processorType, m_deviceIds, m_memoryUsageRatio, cudaCompilerOptions, mklInstructions: m_mklInstructions, enableTensorCore: m_enableTensorCore, m_cudaMemoryAllocatorType, m_elementType);
        }

        public virtual List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, IPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
            => throw new NotImplementedException("RunForwardOnSingleDevice is not implemented.");

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            if (deviceIdIdx < 0 || deviceIdIdx >= DeviceIds.Length)
            {
                throw new ArgumentOutOfRangeException($"Index '{deviceIdIdx}' is out of deviceId range. DeviceId length is '{DeviceIds.Length}'");
            }

            // Create computing graph instance and return it
            return new ComputeGraphTensor(new WeightTensorFactory(), DeviceIds[deviceIdIdx], needBack, saveGPUMemoryMode: m_saveGPUMemoryMode, autoCheckCorruption: m_autoCheckTensorCorruption);
        }
      
        public bool SaveModel(bool createBackupPrevious = false, string suffix = "") => SaveModelImpl(m_modelMetaData, createBackupPrevious, suffix);
        protected virtual bool SaveModelImpl(T model, bool createBackupPrevious = false, string suffix = "") => SaveModelRoutine(model, Model_4_ProtoBufSerializer.Create, createBackupPrevious, suffix);
        protected abstract T LoadModel(string suffix = "");
        protected bool SaveModelRoutine<ProtoBuf_T>(T model, Func<T, ProtoBuf_T> createModel4SerializeFunc, bool createBackupPrevious = false, string suffix = "")
        {
            Logger.WriteLine("Checking if all weights are normal.");
            if (IsWeightsCorrupted())
            {
                throw new WeightsCorruptedException($"The weights has been corrupted. Abort training and please check checkpoint files.");
            }

            string modelFilePath = m_modelFilePath + suffix;
            var fn = Path.GetFullPath(modelFilePath);
            var dir = Path.GetDirectoryName(fn); if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            try
            {
                Logger.WriteLine($"Saving model to '{fn}'");

                if (createBackupPrevious && File.Exists(fn))
                {
                    File.Copy(fn, $"{fn}.bak", true);
                }

                using (var fs = new FileStream(modelFilePath, FileMode.Create, FileAccess.Write))
                {
                    SaveParameters(model);

                    var model_4_serialize = createModel4SerializeFunc(model);
                    ProtoBuf.Serializer.Serialize(fs, model_4_serialize);
                }

                model.ClearWeights();

                return (true);
            }
            catch (Exception ex)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Failed to save model to file. Exception = '{ex.Message}'.");
                Logger.WriteLine(Logger.Level.debug, ConsoleColor.Yellow, $"Call stack = '{ex.StackTrace}'");
                return (false);
            }
        }
		
		// Note(zso): BinaryFormatter deprecated
		#pragma warning disable SYSLIB0011		

        public bool SaveModel_As_BinaryFormatter(bool createBackupPrevious = false, string suffix = "")
        {
            try
            {
                string modelFilePath = m_modelFilePath + suffix;

                Logger.WriteLine($"Saving model to '{modelFilePath}'");

                if (createBackupPrevious && File.Exists(modelFilePath))
                {
                    File.Copy(modelFilePath, $"{modelFilePath}.bak", true);
                }


                BinaryFormatter bf = new BinaryFormatter();
                using (FileStream fs = new FileStream(modelFilePath, FileMode.Create, FileAccess.Write))
                {
                    SaveParameters();
                    // Save model meta data to the stream
                    bf.Serialize(fs, m_modelMetaData);
                    // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be saved to given stream

                }

                m_modelMetaData.ClearWeights();

                return true;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Failed to save model to file. Exception = '{err.Message}', Callstack = '{err.StackTrace}'");
                return false;
            }
        }
        protected T LoadModel_As_BinaryFormatter(Func<T, bool> initializeParametersFunc, string suffix = "")
        {
            Logger.WriteLine($"Loading model from '{m_modelFilePath}'...");
            T model = default;

            BinaryFormatter bf = new BinaryFormatter();
            using (FileStream fs = new FileStream(m_modelFilePath + suffix, FileMode.Open, FileAccess.Read))
            {
                model = bf.Deserialize(fs) as T;

                //Initialize parameters on devices
                initializeParametersFunc(model);

                // Load embedding and weights from given model
                // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be loaded from given stream
                LoadParameters(model);
            }

            return model;
        }

		// Note(zso): BinaryFormatter deprecated
		#pragma warning restore SYSLIB0011

        protected T LoadModelRoutine<ProtoBuf_T>(Func<T, bool> initializeParametersFunc, Func<ProtoBuf_T, T> createModelFunc, string suffix = "")
        {
            Logger.WriteLine($"Loading model from '{m_modelFilePath}'...");
            T model = default;

            try
            {
                using (var fs = new FileStream(m_modelFilePath + suffix, FileMode.Open, FileAccess.Read))
                {
                    var model_4_serialize = ProtoBuf.Serializer.Deserialize<ProtoBuf_T>(fs);
                    model = createModelFunc(model_4_serialize);

                    //Initialize parameters on devices
                    initializeParametersFunc(model);

                    // Load embedding and weights from given model
                    // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be loaded from given stream
                    LoadParameters(model);
                }
            }
            catch (ProtoBuf.ProtoException ex)
            {
                Logger.WriteLine(Logger.Level.warn, $"Failed to load model '{m_modelFilePath + suffix}' as ProtoBuf format. Let's roll back to binary formatter.");
                Logger.WriteLine(Logger.Level.debug, $"Message = '{ex.Message}'");

                model = LoadModel_As_BinaryFormatter(initializeParametersFunc, suffix);
            }


            //For multi-GPUs, copying weights from default device to other all devices
            CopyWeightsFromDefaultDeviceToAllOtherDevices();

            model.ClearWeights();

            Logger.WriteLine(Logger.Level.debug, "Checking if all loaded weights are normal.");

            if (IsWeightsCorrupted())
            {
                throw new WeightsCorruptedException($"The weights has been corrupted. Abort training and please check checkpoint files.");
            }


            return (model);
        }


        internal (MultiProcessorNetworkWrapper<IWeightTensor>, MultiProcessorNetworkWrapper<IWeightTensor>) CreateSrcTgtEmbeddings(IModel modelMetaData, RoundArray<int> raDeviceIds, bool isSrcEmbeddingTrainable, bool isTgtEmbeddingTrainable, float encoderStartLearningRateFactor, float decoderStartLearningRateFactor, DType elementType = DType.Float32)
        {
            MultiProcessorNetworkWrapper<IWeightTensor> srcEmbeddings = null;
            MultiProcessorNetworkWrapper<IWeightTensor> tgtEmbeddings = null;

            if (modelMetaData.SharedEmbeddings)
            {
                Logger.WriteLine(Logger.Level.debug, $"Creating shared embeddings for both source side and target side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");

                srcEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, fanOut: true, name: "SharedEmbeddings", isTrainable: isSrcEmbeddingTrainable, learningRateFactor: encoderStartLearningRateFactor, dtype: elementType), DeviceIds);

                tgtEmbeddings = null;
            }
            else
            {
                Logger.WriteLine(Logger.Level.debug, $"Creating embeddings for source side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");

                srcEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: isSrcEmbeddingTrainable, learningRateFactor: encoderStartLearningRateFactor, dtype: elementType), DeviceIds);

                Logger.WriteLine(Logger.Level.debug, $"Creating embeddings for target side. Shape = '({modelMetaData.TgtVocab.Count} ,{modelMetaData.DecoderEmbeddingDim})'");

                tgtEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.TgtVocab.Count, modelMetaData.DecoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: isTgtEmbeddingTrainable, learningRateFactor: decoderStartLearningRateFactor, dtype: elementType), DeviceIds);
            }

            return (srcEmbeddings, tgtEmbeddings);
        }

        internal MultiProcessorNetworkWrapper<IWeightTensor> CreateTgtEmbeddings(IModel modelMetaData, RoundArray<int> raDeviceIds, bool isTgtEmbeddingTrainable, float decoderStartLearningRateFactor, DType elementType = DType.Float32)
        {
            Logger.WriteLine(Logger.Level.debug, $"Creating embeddings for target side. Shape = '({modelMetaData.TgtVocab.Count} ,{modelMetaData.DecoderEmbeddingDim})'");

            var tgtEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.TgtVocab.Count, modelMetaData.DecoderEmbeddingDim },
                raDeviceIds.GetNextItem(), initType: RandomInitType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: isTgtEmbeddingTrainable, learningRateFactor: decoderStartLearningRateFactor, dtype: elementType), DeviceIds);

            return tgtEmbeddings;
        }


        public void Train(int maxTrainingEpoch, ICorpus<IPairBatch> trainCorpus, ICorpus<IPairBatch>[] validCorpusList, ILearningRate learningRate, Dictionary<int, List<IMetric>> taskId2metrics, IOptimizer optimizer, DecodingOptions decodingOptions)
        {
            Logger.WriteLine("Start to train...");
            LossScaling = m_initLossScaling;

            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpusList, learningRate, optimizer, taskId2metrics, decodingOptions, RunForwardOnSingleDevice);

                // send progress reporting in the form of a percentage value (0-100%)
                var finishedEpochPercent = (int)(100 * (i + 1) / maxTrainingEpoch);
                Logger.WriteLine(Logger.Level.info, $"Finished Epoch Percent: {finishedEpochPercent}%", finishedEpochPercent);
            }

            SaveModel(createBackupPrevious: false, suffix: $".{m_weightsUpdateCount}");
        }

        public void Train(int maxTrainingEpoch, ICorpus<IPairBatch> trainCorpus, ICorpus<IPairBatch>[] validCorpusList, ILearningRate learningRate, IMetric[] metrics, IOptimizer optimizer, DecodingOptions decodingOptions)
        {
            Logger.WriteLine("Start to train...");
            Dictionary<int, List<IMetric>> taskId2metrics = null;
            if (metrics != null)
            {
                taskId2metrics = new Dictionary<int, List<IMetric>>
                {
                    { 0, metrics.ToList() }
                };
            }

            Train(maxTrainingEpoch, trainCorpus, validCorpusList, learningRate, taskId2metrics, optimizer, decodingOptions);
        }


        private void DumpBatchToLogger(List<IPairBatch> batchs)
        {
            foreach (var batch in batchs)
            {
                var srcTokensList = batch.GetSrcTokens();
                var tgtTokensList = batch.GetTgtTokens();

                for (int i = 0; i < srcTokensList.Count; i++)
                {
                    var srcSent = String.Join(" ", srcTokensList[i]);
                    var tgtSent = String.Join(" ", tgtTokensList[i]);

                    Logger.WriteLine(Logger.Level.debug, $"Src = '{srcSent}', Tgt = '{tgtSent}'");
                }
            }
        }

        internal void TrainOneEpoch(int ep, ICorpus<IPairBatch> trainCorpus, ICorpus<IPairBatch>[] validCorpusList, ILearningRate learningRate, IOptimizer solver, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions,
            Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> forwardOnSingleDevice)
        {
            int processedLineInTotal = 0;
            DateTime startDateTime = DateTime.Now;
            DateTime lastCheckpointSaveDateTime = startDateTime;
            double costInTotal = 0.0;
            long srcWordCntsInTotal = 0;
            long tgtWordCntsInTotal = 0;
            double avgCostPerWordInTotal = 0.0;
            int updatesInOneEpoch = 0;
            float lr = 0.0f;
            int contiSuccUpdate = 0;

            Logger.WriteLine(Logger.Level.debug, $"Start to process training corpus.");

            List<IPairBatch> sntPairBatchs = new List<IPairBatch>();

            foreach (var sntPairBatch in trainCorpus)
            {
                sntPairBatchs.Add(sntPairBatch);
                if (sntPairBatchs.Count == m_maxDegressOfParallelism * m_updateFreq)
                {
                    // Copy weights from weights kept in default device to all other devices
                    CopyWeightsFromDefaultDeviceToAllOtherDevices();

                    int batchSplitFactor = 1;
                    bool runNetwordSuccssed = false;

                    while (runNetwordSuccssed == false)
                    {
                        try
                        {
                            (float cost, int sWordCnt, int tWordCnt, int processedLine) = RunNetwork(forwardOnSingleDevice, sntPairBatchs, batchSplitFactor, decodingOptions, true);

                            if (float.IsNaN(cost))
                            {
                                Logger.WriteLine(Logger.Level.warn, "The cost result is Nan,  so we won't update weights at this time.");

                                if (IsWeightsCorrupted())
                                {
                                    throw new WeightsCorruptedException($"The weights has been corrupted. Abort training and please check checkpoint files.");
                                }

                                contiSuccUpdate = 0;
                                break;
                            }

                            processedLineInTotal += processedLine;
                            srcWordCntsInTotal += sWordCnt;
                            tgtWordCntsInTotal += tWordCnt;

                            //Sum up gradients in all devices, and kept it in default device for parameters optmization
                            SumGradientsToTensorsInDefaultDevice();

                            if (IsGradientsCorrupted())
                            {
                                Logger.WriteLine(Logger.Level.warn, $"Gradients is corrupted, so we reduce loss scaling from {LossScaling} to {LossScaling / 2.0f} and skip current batch.");
                                LossScaling = LossScaling * 0.5f;
                                contiSuccUpdate = 0;
                                break;
                            }


                            //Optmize parameters
                            lr = learningRate.GetCurrentLearningRate(ep);
                            if (lr == 0.0f)
                            {
                                throw new ArgumentException($"Learning rate became to 0.0. Try to set larger LearningRateDecaySteps in options to have more training steps.");
                            }

                            List<IWeightTensor> models = GetParametersFromDefaultDevice();

                            m_weightsUpdateCount++;

                            float gradNormFactor = Math.Max(sWordCnt, tWordCnt);
                            if (LossScaling > 0.0f)
                            {
                                gradNormFactor = gradNormFactor / LossScaling;
                            }
                            solver.UpdateWeights(models, gradNormFactor, lr, m_regc, m_weightsUpdateCount);

                            contiSuccUpdate++;
                            if (contiSuccUpdate >= 2000)
                            {
                                if (LossScaling * 2.0f < 32000.0f)
                                {
                                    LossScaling = LossScaling * 2.0f;
                                }

                                contiSuccUpdate = 0;
                            }

                            costInTotal += cost;
                            updatesInOneEpoch++;
                            avgCostPerWordInTotal = costInTotal / updatesInOneEpoch;
                            if (StatusUpdateWatcher != null && m_weightsUpdateCount % 100 == 0)
                            {
                                StatusUpdateWatcher(this, new CostEventArg()
                                {
                                    LearningRate = lr,
                                    AvgCostInTotal = avgCostPerWordInTotal,
                                    Epoch = ep,
                                    Update = m_weightsUpdateCount,
                                    ProcessedSentencesInTotal = processedLineInTotal,
                                    ProcessedWordsInTotal = srcWordCntsInTotal + tgtWordCntsInTotal,
                                    StartDateTime = startDateTime,
                                    LossScaling = LossScaling
                                });
                            }

                            runNetwordSuccssed = true;
                        }
                        catch (AggregateException err)
                        {
                            if (err.InnerExceptions != null)
                            {
                                string oomMessage = string.Empty;
                                bool isOutOfMemException = false;
                                foreach (var excep in err.InnerExceptions)
                                {
                                    if (excep is OutOfMemoryException)
                                    {
                                        GC.Collect();
                                        isOutOfMemException = true;
                                        oomMessage = excep.Message;
                                        break;
                                    }
                                    else
                                    {
                                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Inner Exception: {excep.Message}.");
                                        Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: {excep.StackTrace}");
                                        throw err;
                                    }
                                }

                                if (isOutOfMemException)
                                {
                                    batchSplitFactor = TryToSplitBatchFactor(sntPairBatchs, batchSplitFactor, oomMessage);
                                    if (batchSplitFactor < 0)
                                    {
                                        break;
                                    }
                                }
                                else
                                {
                                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Inner Exception: {err.Message}.");
                                    Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: {err.StackTrace}");
                                    throw err;
                                }
                            }
                            else
                            {
                                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}.");
                                Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: {err.StackTrace}");
                                throw err;
                            }

                        }
                        catch (OutOfMemoryException err)
                        {
                            GC.Collect();
                            batchSplitFactor = TryToSplitBatchFactor(sntPairBatchs, batchSplitFactor, err.Message);
                            if (batchSplitFactor < 0)
                            {
                                break;
                            }
                        }
                        catch (WeightsCorruptedException err)
                        {
                            Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}.");
                            Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: {err.StackTrace}");
                            DumpBatchToLogger(sntPairBatchs);
                            throw;
                        }
                        catch (GradientsCorruptedException err)
                        {
                            Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"We got gradients corruption, ignore current batch: {err.Message}");
                            DumpBatchToLogger(sntPairBatchs);
                            break;
                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}.");
                            Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: {err.StackTrace}");
                            throw;
                        }
                    }

                    if (runNetwordSuccssed == true)
                    {
                        CreateCheckPoint(validCorpusList, taskId2metrics, decodingOptions, forwardOnSingleDevice, avgCostPerWordInTotal);
                    }

                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("e4")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("e4")}");
            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;

            if (EpochEndWatcher != null)
            {
                EpochEndWatcher(this, new CostEventArg()
                {
                    LearningRate = lr,
                    AvgCostInTotal = avgCostPerWordInTotal,
                    Epoch = ep,
                    Update = m_weightsUpdateCount,
                    ProcessedSentencesInTotal = processedLineInTotal,
                    ProcessedWordsInTotal = srcWordCntsInTotal + tgtWordCntsInTotal,
                    StartDateTime = startDateTime
                });

            }
        }

        private int TryToSplitBatchFactor(List<IPairBatch> sntPairBatchs, int batchSplitFactor, string message)
        {
            int maxBatchSize = 0;
            int maxTokenSize = 0;

            foreach (var batch in sntPairBatchs)
            {
                if (maxTokenSize < batch.SrcTokenCount + batch.TgtTokenCount)
                {
                    maxTokenSize = batch.SrcTokenCount + batch.TgtTokenCount;
                }

                if (maxBatchSize < batch.BatchSize)
                {
                    maxBatchSize = batch.BatchSize;
                }
            }

            batchSplitFactor *= 2;

            Logger.WriteLine(Logger.Level.debug, $" {message} Retrying with batch split factor '{batchSplitFactor}'. Max batch size '{maxBatchSize}', Max token size '{maxTokenSize}'");

            if (batchSplitFactor > maxBatchSize)
            {
                Logger.WriteLine(Logger.Level.debug, $"Batch split factor is larger than batch size, so ignore current mini-batch.");

                batchSplitFactor = -1;
            }

            return batchSplitFactor;
        }

        private (float, int, int, int) RunNetwork(Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice, List<IPairBatch> sntPairBatchs, int batchSplitFactor, DecodingOptions decodingOptions, bool isTraining)
        {
            float cost = 0.0f;
            int processedLine = 0;
            int srcWordCnts = 0;
            int tgtWordCnts = 0;

            //Clear gradient over all devices
            ZeroGradientOnAllDevices();

            int currBatchIdx = -1;

            // Run forward and backward on all available processors
            Parallel.For(0, m_maxDegressOfParallelism, i =>
            {
                int deviceIdx = i % m_deviceIds.Length;
                int loclCurrBatchIdx = Interlocked.Increment(ref currBatchIdx);
                while (loclCurrBatchIdx < sntPairBatchs.Count)
                {
                    try
                    {

                        var sntPairBatch_i = sntPairBatchs[loclCurrBatchIdx];
                        int batchSegSize = sntPairBatch_i.BatchSize / batchSplitFactor;
                        if (batchSegSize > 0)
                        {
                            for (int k = 0; k < batchSplitFactor; k++)
                            {
                                var sntPairBatch = sntPairBatch_i.GetRange(k * batchSegSize, batchSegSize);

                                List<NetworkResult> nrs;
                                // Create a new computing graph instance
                                using (IComputeGraph computeGraph_deviceIdx = CreateComputGraph(deviceIdx))
                                {
                                    // Run forward part
                                    using (IComputeGraph g = computeGraph_deviceIdx.CreateSubGraph($"Forward_{deviceIdx}"))
                                    {
                                        nrs = ForwardOnSingleDevice(g, sntPairBatch, decodingOptions, isTraining);
                                    }

                                    // Run backward part and compute gradients
                                    computeGraph_deviceIdx.Backward();
                                }

                                GC.Collect();

                                lock (locker)
                                {
                                    foreach (var nr in nrs)
                                    {
                                        cost += nr.Cost;
                                    }

                                    srcWordCnts += sntPairBatch_i.SrcTokenCount;
                                    tgtWordCnts += sntPairBatch_i.TgtTokenCount;
                                    processedLine += batchSegSize;
                                }
                            }
                        }

                        int remainBatchSegSize = sntPairBatch_i.BatchSize % batchSplitFactor;
                        if (remainBatchSegSize > 0)
                        {
                            var sntPairBatch = sntPairBatch_i.GetRange(sntPairBatch_i.BatchSize - remainBatchSegSize, remainBatchSegSize);

                            List<NetworkResult> nrs;
                             // Create a new computing graph instance
                             using (IComputeGraph computeGraph_deviceIdx = CreateComputGraph(deviceIdx))
                            {
                                 // Run forward part
                                 nrs = ForwardOnSingleDevice(computeGraph_deviceIdx, sntPairBatch, decodingOptions, isTraining);
                                 // Run backward part and compute gradients
                                 computeGraph_deviceIdx.Backward();
                            }

                             GC.Collect();

                            lock (locker)
                            {
                                foreach (var nr in nrs)
                                {
                                    cost += nr.Cost;
                                }

                                srcWordCnts += sntPairBatch_i.SrcTokenCount;
                                tgtWordCnts += sntPairBatch_i.TgtTokenCount;
                                processedLine += batchSegSize;
                            }

                        }
                    }
                    catch (OutOfMemoryException err)
                    {
                        GC.Collect();
                        throw err;
                    }
                    catch (Exception err)
                    {
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: '{err.Message}'");
                        Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");

                        throw;
                    }

                    loclCurrBatchIdx = Interlocked.Increment(ref currBatchIdx);
                }
            });

            return (cost / processedLine, srcWordCnts, tgtWordCnts, processedLine);
        }

        private void CreateCheckPoint(ICorpus<IPairBatch>[] validCorpusList, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions, Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> forwardOnSingleDevice, double avgCostPerWordInTotal)
        {
            // We start to run validation after {startToRunValidAfterUpdates}
            if (m_weightsUpdateCount >= m_startToRunValidAfterUpdates && m_weightsUpdateCount % m_runValidEveryUpdates == 0)
            {
                if (validCorpusList != null && validCorpusList.Length > 0)
                {
                    ReleaseGradientOnAllDevices();

                    // The valid corpus is provided, so evaluate the model.
                    for (int i = 0; i < validCorpusList.Length; i++)
                    {
                        var validCorpus = validCorpusList[i];
                        var betterResult = RunValid(validCorpus, forwardOnSingleDevice, taskId2metrics, decodingOptions, outputToFile: true, prefixName: validCorpus.CorpusName);

                        if ((i == 0 && betterResult == true) || File.Exists(m_modelFilePath) == false)
                        {
                            //---SaveModel_As_BinaryFormatter();
                            SaveModel(createBackupPrevious: true);
                        }
                    }
                }
                else if (m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal || File.Exists(m_modelFilePath) == false)
                {
                    // We don't have valid corpus, so if we could have lower cost, save the model
                    //---SaveModel_As_BinaryFormatter();
                    SaveModel(createBackupPrevious: true);
                }
            }

            if (m_saveModelEveryUpdates > 0 && m_weightsUpdateCount % m_saveModelEveryUpdates == 0 && m_weightsUpdateCount > 0)
            {
                SaveModel(createBackupPrevious: false, suffix: $".{m_weightsUpdateCount}");
            }
        }


        private static List<NetworkResult> MergeResults(SortedDictionary<int, List<NetworkResult>> batchId2Results)
        {
            List<NetworkResult> rs = new List<NetworkResult>();


            foreach (var pair in batchId2Results)
            {
                var tasks = pair.Value;
                if (rs.Count == 0)
                {
                    for (int i = 0; i < tasks.Count; i++)
                    {
                        NetworkResult nr = new NetworkResult
                        {
                            Output = new List<List<List<string>>>()
                        };
                        rs.Add(nr);
                    }
                }


                for (int i = 0; i < tasks.Count; i++)
                {
                    rs[i].AppendResult(tasks[i]);
                }
            }

            return rs;
        }


        internal List<NetworkResult> RunTest(IPairBatch sntPairBatch, DecodingOptions decodingOptions, Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice)
        {
            if (sntPairBatch is null)
            {
                throw new ArgumentNullException(nameof(sntPairBatch));
            }

            if (ForwardOnSingleDevice is null)
            {
                throw new ArgumentNullException(nameof(ForwardOnSingleDevice));
            }

            try
            {
                SortedDictionary<int, List<NetworkResult>> batchId2Result = new SortedDictionary<int, List<NetworkResult>>();

                int dataSizePerGPU = sntPairBatch.BatchSize / m_maxDegressOfParallelism;
                int dataSizePerGPUMod = sntPairBatch.BatchSize % m_maxDegressOfParallelism;

                if (dataSizePerGPU > 0)
                {
                    Parallel.For(0, m_maxDegressOfParallelism, i =>
                    {
                        int deviceIdx = i % m_deviceIds.Length;
                        try
                        {
                            var spb = sntPairBatch.GetRange(deviceIdx * dataSizePerGPU, dataSizePerGPU);

                            List<NetworkResult> nrs = null;
                             // Create a new computing graph instance
                             using (IComputeGraph computeGraph = CreateComputGraph(deviceIdx, needBack: false))
                            {
                                 // Run forward part
                                 nrs = ForwardOnSingleDevice(computeGraph, spb, decodingOptions, false);
                            }

                            GC.Collect();

                            lock (locker)
                            {
                                batchId2Result.Add(deviceIdx, nrs);
                            }

                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, $"Test error at processor '{deviceIdx}'. Exception = '{err.Message}'.");
                            Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                            throw;
                        }
                    });
                }

                if (dataSizePerGPUMod > 0)
                {
                    var spb = sntPairBatch.GetRange(m_maxDegressOfParallelism * dataSizePerGPU, dataSizePerGPUMod);

                    List<NetworkResult> nrs2 = null;
                    // Create a new computing graph instance
                    using (IComputeGraph computeGraph = CreateComputGraph(0, needBack: false))
                    {
                        // Run forward part
                        nrs2 = ForwardOnSingleDevice(computeGraph, spb, decodingOptions, false);
                    }

                    GC.Collect();

                    lock (locker)
                    {
                        batchId2Result.Add(m_maxDegressOfParallelism, nrs2);
                    }

                }

                List<NetworkResult> nrs = MergeResults(batchId2Result);

                return nrs;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                throw;
            }
        }


        public List<NetworkResult> Test<X>(List<List<string>> inputTokens, List<List<string>> outputTokens, DecodingOptions decodingOptions) where X : IPairBatch, new()
        {
            X spb = new X();
            spb.CreateBatch(inputTokens, outputTokens);
            var nrs = RunTest(spb, decodingOptions, RunForwardOnSingleDevice);

            return nrs;
        }

        public void Test<X>(string inputTestFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null) where X : IPairBatch, new()
        {
            SntBatchStreamReader<X> reader = new SntBatchStreamReader<X>(inputTestFile, batchSize, decodingOptions.MaxSrcSentLength, srcSpmPath);
            SntBatchStreamWriter writer = new SntBatchStreamWriter(outputFile, tgtSpmPath, outputAlignmentFile);
            RunTest<X>(reader, writer, decodingOptions, RunForwardOnSingleDevice);
        }

        public void Test<X>(string inputTestFile, string inputPromptFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null) where X : IPairBatch, new()
        {
            SntPairBatchStreamReader<X> reader = new SntPairBatchStreamReader<X>(inputTestFile, inputPromptFile, batchSize, decodingOptions.MaxSrcSentLength, srcSpmPath, tgtSpmPath);
            SntBatchStreamWriter writer = new SntBatchStreamWriter(outputFile, tgtSpmPath, outputAlignmentFile);
            RunTest<X>(reader, writer, decodingOptions, RunForwardOnSingleDevice);
        }


        internal void RunTest<X>(IBatchStreamReader<X> reader, SntBatchStreamWriter writer, DecodingOptions decodingOptions, Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice) where X : IPairBatch, new()
        {
            if (ForwardOnSingleDevice is null)
            {
                throw new ArgumentNullException(nameof(ForwardOnSingleDevice));
            }

            bool runningGoodSoFar = true;
            try
            {
                Parallel.For(0, m_maxDegressOfParallelism, i =>
                {
                    int deviceIdx = i % m_deviceIds.Length;
                    try
                    {
                        while (runningGoodSoFar)
                        {
                            (var idx, var spb) = reader.GetNextBatch();
                            if (idx < 0)
                            {
                                break;
                            }

                            List<NetworkResult> nrs = null;
                             // Create a new computing graph instance
                             using (IComputeGraph computeGraph = CreateComputGraph(deviceIdx, needBack: false))
                            {
                                 // Run forward part
                                 nrs = ForwardOnSingleDevice(computeGraph, spb, decodingOptions, false);
                            }

                            GC.Collect();

                            writer.WriteResults(idx, nrs);
                        }
                    }
                    catch (Exception err)
                    {
                        runningGoodSoFar = false;
                        Logger.WriteLine(Logger.Level.err, $"Test error at processor '{deviceIdx}'. Exception = '{err.Message}'.");
                        Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                        throw;
                    }
                });

                writer.Close();
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call Stack = '{err.StackTrace}'");
                throw;
            }
        }


        public void Valid(ICorpus<ISntPairBatch> validCorpus, List<IMetric> metrics, DecodingOptions decodingOptions)
        {
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>
            {
                { 0, metrics }
            };
            RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, decodingOptions, true);
        }

        public void Valid(ICorpus<ISntPairBatch> validCorpus, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions) => RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, decodingOptions, true);

        /// <summary>
        /// Evaluate the quality of model on valid corpus.
        /// </summary>
        /// <param name="validCorpus">valid corpus to measure the quality of model</param>
        /// <param name="RunNetwork">The network to run on specific device</param>
        /// <param name="metrics">A set of metrics. The first one is the primary metric</param>
        /// <param name="outputToFile">It indicates if valid corpus and results should be dumped to files</param>
        /// <returns>true if we get a better result on primary metric, otherwise, false</returns>
        internal bool RunValid(ICorpus<IPairBatch> validCorpus, Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> RunNetwork, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions, bool outputToFile = false, string prefixName = "valid")
        {
            double bestPrimaryScore = 0.0;
            if (m_bestPrimaryScoreDict.ContainsKey(prefixName) == false)
            {
                m_bestPrimaryScoreDict.Add(prefixName, 0.0);
            }
            else
            {
                bestPrimaryScore = m_bestPrimaryScoreDict[prefixName];
            }

            // Clear inner status of each metrics
            foreach (var pair in taskId2metrics)
            {
                foreach (IMetric metric in pair.Value)
                {
                    metric.ClearStatus();
                }
            }

            string srcFileName = $"{prefixName}_src.txt";
            string refFileName = $"{prefixName}_ref.txt";
            string hypFileName = $"{prefixName}_hyp.txt";
            if (outputToFile)
            {
                if (File.Exists(srcFileName))
                {
                    File.Delete(srcFileName);
                }

                if (File.Exists(refFileName))
                {
                    File.Delete(refFileName);
                }

                if (File.Exists(hypFileName))
                {
                    File.Delete(hypFileName);
                }
            }

            CopyWeightsFromDefaultDeviceToAllOtherDevices();

            List<IPairBatch> sntPairBatchs = new List<IPairBatch>();
            foreach (var item in validCorpus)
            {
                sntPairBatchs.Add(item);
                if (sntPairBatchs.Count == DeviceIds.Length)
                {
                    RunValidParallel(RunNetwork, taskId2metrics, decodingOptions, prefixName, outputToFile, sntPairBatchs);
                    sntPairBatchs.Clear();
                }
            }

            if (sntPairBatchs.Count > 0)
            {
                RunValidParallel(RunNetwork, taskId2metrics, decodingOptions, prefixName, outputToFile, sntPairBatchs);
            }

            bool betterModel = false;
            if (taskId2metrics.Count > 0)
            {
                StringBuilder sb = new StringBuilder();
                List<IMetric> metricList = new List<IMetric>();

                foreach (var pair in taskId2metrics) // Run metrics for each task
                {
                    int taskId = pair.Key;
                    List<IMetric> metrics = pair.Value;
                    metricList.AddRange(metrics);

                    sb.AppendLine($"Metrics result on task '{taskId}' on data set '{prefixName}':");
                    foreach (IMetric metric in metrics)
                    {
                        sb.AppendLine($"{metric.Name} = {metric.GetScoreStr()}");
                    }

                    if (metrics[0].GetPrimaryScore() > bestPrimaryScore && taskId == m_primaryTaskId) // The first metric in the primary task is the primary metric
                    {
                        if (bestPrimaryScore > 0.0f)
                        {
                            sb.AppendLine($"We got a better primary metric '{metrics[0].Name}' score '{metrics[0].GetPrimaryScore():F}' on the primary task '{taskId}' and data set '{prefixName}'. The previous score is '{bestPrimaryScore:F}'");
                        }

                        //We have a better primary score on valid set
                        bestPrimaryScore = metrics[0].GetPrimaryScore();
                        m_bestPrimaryScoreDict[prefixName] = bestPrimaryScore;
                        betterModel = true;
                    }
                }

                if (EvaluationWatcher != null)
                {
                    EvaluationWatcher(this, new EvaluationEventArg()
                    {
                        Title = $"Evaluation result for model '{m_modelFilePath}' on test set '{prefixName}'",
                        Message = sb.ToString(),
                        Metrics = metricList,
                        BetterModel = betterModel,
                        Color = ConsoleColor.Green
                    });
                }
            }

            return betterModel;
        }

        private void RunValidParallel(Func<IComputeGraph, IPairBatch, DecodingOptions, bool, List<NetworkResult>> runNetwork, Dictionary<int, List<IMetric>> metrics, DecodingOptions decodingOptions, string taskPrefixName, bool outputToFile, List<IPairBatch> sntPairBatchs)
        {
            string srcFileName = $"{taskPrefixName}_src.txt";
            string refFileName = $"{taskPrefixName}_ref.txt";
            string hypFileName = $"{taskPrefixName}_hyp.txt";

            // Run forward on all available processors
            Parallel.For(0, m_maxDegressOfParallelism, i =>
            {
                int deviceIdx = i % m_deviceIds.Length;
                try
                {
                    if (deviceIdx >= sntPairBatchs.Count)
                    {
                        return;
                    }

                    var sntPairBatch = sntPairBatchs[deviceIdx];
                    var sntPairBatchForValid = sntPairBatch.CloneSrcTokens();

                    // Create a new computing graph instance
                    List<NetworkResult> nrs;
                    using (IComputeGraph computeGraph = CreateComputGraph(deviceIdx, needBack: false))
                    {
                        // Run forward part
                        nrs = runNetwork(computeGraph, sntPairBatchForValid, decodingOptions, false);
                    }

                    GC.Collect();

                    lock (locker)
                    {
                        string[] newSrcSnts = new string[sntPairBatch.BatchSize];
                        string[] newRefSnts = new string[sntPairBatch.BatchSize];
                        string[] newHypSnts = new string[sntPairBatch.BatchSize];


                        for (int k = 0; k < nrs.Count; k++)
                        {
                            var hypTkns = nrs[k].Output[0];
                            var refTkns = sntPairBatch.GetTgtTokens();
                            var srcTkns = sntPairBatch.GetSrcTokens();

                            for (int j = 0; j < hypTkns.Count; j++)
                            {
                                foreach (IMetric metric in metrics[k])
                                {
                                    if (j < 0 || j >= refTkns.Count)
                                    {
                                        throw new InvalidDataException($"Ref token only has '{refTkns.Count}' batch, however, it try to access batch '{j}'. Hyp token has '{hypTkns.Count}' tokens, Batch Size = '{sntPairBatch.BatchSize}'");
                                    }

                                    if (j < 0 || j >= hypTkns.Count)
                                    {
                                        throw new InvalidDataException($"Hyp token only has '{hypTkns.Count}' batch, however, it try to access batch '{j}'. Ref token has '{refTkns.Count}' tokens, Batch Size = '{sntPairBatch.BatchSize}'");
                                    }

                                    try
                                    {
                                        metric.Evaluate(new List<List<string>>() { refTkns[j] }, hypTkns[j]);
                                    }
                                    catch (Exception err)
                                    {
                                        Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}', Ref = '{string.Join(" ", refTkns[j])}' Hyp = '{string.Join(" ", hypTkns[j])}', TaskId = '{k}'");
                                        throw;
                                    }
                                }
                            }

                            if (outputToFile)
                            {
                                for (int j = 0; j < srcTkns.Count; j++)
                                {
                                    if (k == 0)
                                    {
                                        newSrcSnts[j] = string.Join(" ", srcTkns[j]);
                                    }

                                    newRefSnts[j] += string.Join(" ", refTkns[j]) + "\t";
                                    newHypSnts[j] += string.Join(" ", hypTkns[j]) + "\t";
                                }
                            }
                        }

                        if (outputToFile)
                        {
                            File.AppendAllLines(srcFileName, newSrcSnts);
                            File.AppendAllLines(refFileName, newRefSnts);
                            File.AppendAllLines(hypFileName, newHypSnts);							
                        }

                    }
                }
                catch (OutOfMemoryException err)
                {
                    GC.Collect(); // Collect unused tensor objects and free GPU memory

                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Skip current batch for validation due to {err.Message}");
                }
                catch (Exception err)
                {
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: '{err.Message}'.");
                    Logger.WriteLine(Logger.Level.debug, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");
                }
            });
        }

        internal virtual void SaveParameters()
        {
            m_modelMetaData.ClearWeights();

            RegisterTrainableParameters(this);

            HashSet<IMultiProcessorNetworkWrapper> setNetworkWrapper = new HashSet<IMultiProcessorNetworkWrapper>();
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                if (setNetworkWrapper.Contains(pair.Value) == false)
                {
                    setNetworkWrapper.Add(pair.Value);
                    pair.Value.Save(m_modelMetaData);
                }
                else
                {
                    throw new ArgumentException($"Failed to save parameter due to duplicated parameter name '{pair.Value}'");
                }
            }
        }
        internal virtual void LoadParameters()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                Logger.WriteLine(Logger.Level.debug, $"Loading parameter '{pair.Key}'");

                pair.Value.Load(m_modelMetaData);
            }
        }

        protected virtual void SaveParameters(IModel model)
        {
            model.ClearWeights();

            RegisterTrainableParameters(this);

            var setNetworkWrapper = new HashSet<IMultiProcessorNetworkWrapper>(m_name2network.Count);
            foreach (IMultiProcessorNetworkWrapper mpnw in m_name2network.Values)
            {
                if (setNetworkWrapper.Add(mpnw))
                {
                    mpnw.Save(model);
                }
                else
                {
                    throw new ArgumentException($"Failed to save parameter due to duplicated network wrapper.'");
                }
            }
        }
        protected virtual void LoadParameters(IModel model)
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> p in m_name2network)
            {
                var name = p.Key;
                var mpnw = p.Value;

                Logger.WriteLine(Logger.Level.debug, $"Loading parameter '{name}'");

                mpnw.Load(model);
            }
        }


        internal bool IsWeightsCorrupted()
        {
            var weights = GetParametersFromDefaultDevice();

            foreach (var weight in weights)
            {
                if (weight.IsWeightsCorrupted())
                {
                    Logger.WriteLine(Logger.Level.err, $"Weight '{weight.Name}' is corrupted.");
                    return true;
                }
            }

            return false;
        }

        internal bool IsGradientsCorrupted()
        {
            var weights = GetParametersFromDefaultDevice();

            foreach (var weight in weights)
            {
                if (weight.IsGradientCorrupted())
                {
                    Logger.WriteLine(Logger.Level.err, $"Gradient '{weight.Name}' is corrupted.");
                    return true;
                }
            }

            return false;
        }
        /// <summary>
        /// Copy weights from default device to all other devices
        /// </summary>
        internal void CopyWeightsFromDefaultDeviceToAllOtherDevices()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                pair.Value.SyncWeights();
            }
        }

        /// <summary>
        /// Sum up gradients in all devices and keep them in the default device
        /// </summary>
        internal void SumGradientsToTensorsInDefaultDevice()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                pair.Value.SumGradientsToNetworkOnDefaultDevice();
            }
        }

        internal List<IWeightTensor> GetParametersFromDefaultDevice()
        {
            RegisterTrainableParameters(this);
            List<IWeightTensor> result = new List<IWeightTensor>();
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                result.AddRange(pair.Value.GetWeightsOnDefaultDevice());
            }

            return result;
        }

        internal void ZeroGradientOnAllDevices()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                pair.Value.ZeroGradientsOnAllDevices();
            }
        }

        internal void ReleaseGradientOnAllDevices()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                pair.Value.ReleaseGradientsOnAllDevices();
            }
        }

        internal void RegisterTrainableParameters(object obj)
        {
            if (m_name2network != null)
            {
                return;
            }
            Logger.WriteLine(Logger.Level.debug, $"Registering trainable parameters.");

            m_name2network = new SortedList<string, IMultiProcessorNetworkWrapper>();

            foreach (FieldInfo childFieldInfo in obj.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                object childValue = childFieldInfo.GetValue(obj);
                string name = childFieldInfo.Name;
                Register(childValue, name);
            }
            foreach (PropertyInfo childPropertyInfo in obj.GetType().GetProperties(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                object childValue = childPropertyInfo.GetValue(obj);
                string name = childPropertyInfo.Name;
                Register(childValue, name);
            }
        }

        private void Register(object childValue, string name)
        {
            if (childValue is IMultiProcessorNetworkWrapper networks)
            {
                m_name2network.Add(name, networks);

                Logger.WriteLine(Logger.Level.debug, $"Register network '{name}'");
            }

            if (childValue is IMultiProcessorNetworkWrapper[] networksArray)
            {
                int idx = 0;
                foreach (var network in networksArray)
                {
                    string name2 = $"{name}_{idx}";
                    m_name2network.Add(name2, network);

                    Logger.WriteLine(Logger.Level.debug, $"Register network '{name2}'");

                    idx++;
                }
            }
        }
    }
}
