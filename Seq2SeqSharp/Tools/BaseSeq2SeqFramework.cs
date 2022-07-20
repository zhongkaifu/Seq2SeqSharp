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
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Metrics;
using Seq2SeqSharp.Models;
using Seq2SeqSharp.Optimizer;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Tools
{
    public class NetworkResult
    {
        public float Cost;
        public List<List<List<string>>> Output; // (beam_size, batch_size, seq_len)
        public List<List<List<int>>> Alignments; // (beam_size, batch_size, seq_len)
        public List<List<List<float>>> AlignmentScores; // (beam_size, batch_size, seq_len)

        public NetworkResult()
        {
            Output = null;
            Alignments = null;
            AlignmentScores = null;

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
        private int m_weightsUpdateCount = 0;
        private double m_avgCostPerWordInTotalInLastEpoch = 10000.0;
        private Dictionary<string, double> m_bestPrimaryScoreDict = new Dictionary<string, double>();
        private readonly int m_primaryTaskId = 0;
        private readonly object locker = new object();
        private SortedList<string, IMultiProcessorNetworkWrapper> m_name2network;
        private int m_updateFreq = 1;
        private int m_startToRunValidAfterUpdates = 20000;
        private int m_runValidEveryUpdates = 10000;
        private int m_maxDegressOfParallelism = 1;

        public BaseSeq2SeqFramework(string deviceIds, ProcessorTypeEnums processorType, string modelFilePath, float memoryUsageRatio = 0.9f, 
            string compilerOptions = null, int runValidEveryUpdates = 10000, int primaryTaskId = 0, int updateFreq = 1, int startToRunValidAfterUpdates = 0,
            int maxDegressOfParallelism = 1)
        {
            m_deviceIds = deviceIds.Split(',').Select(x => int.Parse(x)).ToArray();
            string[] cudaCompilerOptions = compilerOptions.IsNullOrEmpty() ? null : compilerOptions.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            m_modelFilePath = modelFilePath;
            TensorAllocator.InitDevices(processorType, m_deviceIds, memoryUsageRatio, cudaCompilerOptions);

            m_primaryTaskId = primaryTaskId;
            m_updateFreq = updateFreq;
            m_startToRunValidAfterUpdates = startToRunValidAfterUpdates;
            m_runValidEveryUpdates = runValidEveryUpdates;
            m_maxDegressOfParallelism = maxDegressOfParallelism;
        }

        public virtual List<NetworkResult> RunForwardOnSingleDevice(IComputeGraph computeGraph, ISntPairBatch sntPairBatch, DecodingOptions decodingOptions, bool isTraining)
            => throw new NotImplementedException("RunForwardOnSingleDevice is not implemented.");

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            if (deviceIdIdx < 0 || deviceIdIdx >= DeviceIds.Length)
            {
                throw new ArgumentOutOfRangeException($"Index '{deviceIdIdx}' is out of deviceId range. DeviceId length is '{DeviceIds.Length}'");
            }

            // Create computing graph instance and return it
            return new ComputeGraphTensor(new WeightTensorFactory(), DeviceIds[deviceIdIdx], needBack);
        }
      
        protected T LoadModelImpl_WITH_CONVERT(Func<T, bool> initializeParametersFunc)
        {
                return (LoadModelImpl());
        }

        public bool SaveModel(bool createBackupPrevious = false, string suffix = "") => SaveModelImpl(m_modelMetaData, createBackupPrevious, suffix);
        protected virtual bool SaveModelImpl(T model, bool createBackupPrevious = false, string suffix = "") => SaveModelRoutine(model, Model_4_ProtoBufSerializer.Create, createBackupPrevious, suffix);
        protected abstract T LoadModelImpl();
        protected bool SaveModelRoutine<ProtoBuf_T>(T model, Func<T, ProtoBuf_T> createModel4SerializeFunc, bool createBackupPrevious = false, string suffix = "")
        {
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
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Failed to save model to file. Exception = '{ex.Message}', Call stack = '{ex.StackTrace}'");
                return (false);
            }
        }
        protected T LoadModelRoutine<ProtoBuf_T>(Func<T, bool> initializeParametersFunc, Func<ProtoBuf_T, T> createModelFunc)
        {
            Logger.WriteLine($"Loading model from '{m_modelFilePath}'...");
            T model = default;

            using (var fs = new FileStream(m_modelFilePath, FileMode.Open, FileAccess.Read))
            {
                var model_4_serialize = ProtoBuf.Serializer.Deserialize<ProtoBuf_T>(fs);
                model = createModelFunc(model_4_serialize);

                //Initialize parameters on devices
                initializeParametersFunc(model);

                // Load embedding and weights from given model
                // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be loaded from given stream
                LoadParameters(model);
            }

            //For multi-GPUs, copying weights from default device to other all devices
            CopyWeightsFromDefaultDeviceToAllOtherDevices();

            model.ClearWeights();

            return (model);
        }


        internal (MultiProcessorNetworkWrapper<IWeightTensor>, MultiProcessorNetworkWrapper<IWeightTensor>) CreateSrcTgtEmbeddings(IModel modelMetaData, RoundArray<int> raDeviceIds, bool isSrcEmbeddingTrainable, bool isTgtEmbeddingTrainable, float encoderStartLearningRateFactor, float decoderStartLearningRateFactor)
        {
            MultiProcessorNetworkWrapper<IWeightTensor> srcEmbeddings = null;
            MultiProcessorNetworkWrapper<IWeightTensor> tgtEmbeddings = null;

            if (modelMetaData.SharedEmbeddings)
            {
                Logger.WriteLine($"Creating shared embeddings for both source side and target side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
                srcEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SharedEmbeddings", isTrainable: isSrcEmbeddingTrainable, learningRateFactor: encoderStartLearningRateFactor), DeviceIds);

                tgtEmbeddings = srcEmbeddings;
            }
            else
            {
                Logger.WriteLine($"Creating embeddings for source side. Shape = '({modelMetaData.SrcVocab.Count} ,{modelMetaData.EncoderEmbeddingDim})'");
                srcEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.SrcVocab.Count, modelMetaData.EncoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "SrcEmbeddings", isTrainable: isSrcEmbeddingTrainable, learningRateFactor: encoderStartLearningRateFactor), DeviceIds);

                Logger.WriteLine($"Creating embeddings for target side. Shape = '({modelMetaData.TgtVocab.Count} ,{modelMetaData.DecoderEmbeddingDim})'");
                tgtEmbeddings = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.TgtVocab.Count, modelMetaData.DecoderEmbeddingDim },
                    raDeviceIds.GetNextItem(), normType: NormType.Uniform, fanOut: true, name: "TgtEmbeddings", isTrainable: isTgtEmbeddingTrainable, learningRateFactor: decoderStartLearningRateFactor), DeviceIds);
            }

            return (srcEmbeddings, tgtEmbeddings);
        }

        public void Train(int maxTrainingEpoch, IParallelCorpus<ISntPairBatch> trainCorpus, IParallelCorpus<ISntPairBatch>[] validCorpusList, ILearningRate learningRate, Dictionary<int, List<IMetric>> taskId2metrics, IOptimizer optimizer, DecodingOptions decodingOptions)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, trainCorpus, validCorpusList, learningRate, optimizer, taskId2metrics, decodingOptions, RunForwardOnSingleDevice);
            }
        }

        public void Train(int maxTrainingEpoch, IParallelCorpus<ISntPairBatch> trainCorpus, IParallelCorpus<ISntPairBatch>[] validCorpusList, ILearningRate learningRate, List<IMetric> metrics, IOptimizer optimizer, DecodingOptions decodingOptions)
        {
            Logger.WriteLine("Start to train...");
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>
            {
                { 0, metrics }
            };

            Train(maxTrainingEpoch, trainCorpus, validCorpusList, learningRate, taskId2metrics, optimizer, decodingOptions);
        }

        internal void TrainOneEpoch(int ep, IParallelCorpus<ISntPairBatch> trainCorpus, IParallelCorpus<ISntPairBatch>[] validCorpusList, ILearningRate learningRate, IOptimizer solver, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions,
            Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> forwardOnSingleDevice)
        {
            int processedLineInTotal = 0;
            DateTime startDateTime = DateTime.Now;
            double costInTotal = 0.0;
            long srcWordCntsInTotal = 0;
            long tgtWordCntsInTotal = 0;
            double avgCostPerWordInTotal = 0.0;
            int updatesInOneEpoch = 0;
            float lr = 0.0f;

            Logger.WriteLine($"Start to process training corpus.");
            List<ISntPairBatch> sntPairBatchs = new List<ISntPairBatch>();

            foreach (ISntPairBatch sntPairBatch in trainCorpus)
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
                            processedLineInTotal += processedLine;
                            srcWordCntsInTotal += sWordCnt;
                            tgtWordCntsInTotal += tWordCnt;

                            //Sum up gradients in all devices, and kept it in default device for parameters optmization
                            SumGradientsToTensorsInDefaultDevice();

                            //Optmize parameters
                            lr = learningRate.GetCurrentLearningRate();
                            List<IWeightTensor> models = GetParametersFromDefaultDevice();

                            m_weightsUpdateCount++;
                            solver.UpdateWeights(models, processedLine, lr, m_regc, m_weightsUpdateCount);

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
                                    StartDateTime = startDateTime
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
                                bool isArithmeticException = false;
                                foreach (var excep in err.InnerExceptions)
                                {
                                    if (excep is OutOfMemoryException)
                                    {
                                        GC.Collect();
                                        isOutOfMemException = true;
                                        oomMessage = excep.Message;
                                        break;
                                    }
                                    else if (excep is ArithmeticException)
                                    {
                                        isArithmeticException = true;
                                        oomMessage = excep.Message;
                                        break;
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
                                else if (isArithmeticException)
                                {
                                    Logger.WriteLine($"Arithmetic exception: '{err.Message}'");
                                    break;
                                }
                                else
                                {
                                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
                                    throw err;
                                }
                            }
                            else
                            {
                                Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
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
                        catch (ArithmeticException err)
                        {
                            Logger.WriteLine($"Arithmetic exception: '{err.Message}'");
                            break;
                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: {err.Message}, Call stack: {err.StackTrace}");
                            throw;
                        }
                    }

                    CreateCheckPoint(validCorpusList, taskId2metrics, decodingOptions, forwardOnSingleDevice, avgCostPerWordInTotal);
                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal:F6}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch:F6}");
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

        private int TryToSplitBatchFactor(List<ISntPairBatch> sntPairBatchs, int batchSplitFactor, string message)
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
            Logger.WriteLine($" {message} Retrying with batch split factor '{batchSplitFactor}'. Max batch size '{maxBatchSize}', Max token size '{maxTokenSize}'");

            if (batchSplitFactor > maxBatchSize)
            {
                Logger.WriteLine($"Batch split factor is larger than batch size, so ignore current mini-batch.");
                batchSplitFactor = -1;
            }

            return batchSplitFactor;
        }

        private (float, int, int, int) RunNetwork(Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice, List<ISntPairBatch> sntPairBatchs, int batchSplitFactor, DecodingOptions decodingOptions, bool isTraining)
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

                        ISntPairBatch sntPairBatch_i = sntPairBatchs[loclCurrBatchIdx];
                        int batchSegSize = sntPairBatch_i.BatchSize / batchSplitFactor;
                        if (batchSegSize > 0)
                        {
                            for (int k = 0; k < batchSplitFactor; k++)
                            {
                                ISntPairBatch sntPairBatch = sntPairBatch_i.GetRange(k * batchSegSize, batchSegSize);

                                List<NetworkResult> nrs;
                                 // Create a new computing graph instance
                                 using (IComputeGraph computeGraph_deviceIdx = CreateComputGraph(deviceIdx))
                                {
                                     // Run forward part
                                     nrs = ForwardOnSingleDevice(computeGraph_deviceIdx, sntPairBatch, decodingOptions, isTraining);
                                     // Run backward part and compute gradients
                                     computeGraph_deviceIdx.Backward();
                                }

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
                            ISntPairBatch sntPairBatch = sntPairBatch_i.GetRange(sntPairBatch_i.BatchSize - remainBatchSegSize, remainBatchSegSize);

                            List<NetworkResult> nrs;
                             // Create a new computing graph instance
                             using (IComputeGraph computeGraph_deviceIdx = CreateComputGraph(deviceIdx))
                            {
                                 // Run forward part
                                 nrs = ForwardOnSingleDevice(computeGraph_deviceIdx, sntPairBatch, decodingOptions, isTraining);
                                 // Run backward part and compute gradients
                                 computeGraph_deviceIdx.Backward();
                            }

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
                        Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");

                        throw;
                    }

                    loclCurrBatchIdx = Interlocked.Increment(ref currBatchIdx);
                }
            });

            return (cost / processedLine, srcWordCnts, tgtWordCnts, processedLine);
        }

        private void CreateCheckPoint(IParallelCorpus<ISntPairBatch>[] validCorpusList, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions, Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> forwardOnSingleDevice, double avgCostPerWordInTotal)
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


        internal List<NetworkResult> RunTest(ISntPairBatch sntPairBatch, DecodingOptions decodingOptions, Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice)
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
                            ISntPairBatch spb = sntPairBatch.GetRange(deviceIdx * dataSizePerGPU, dataSizePerGPU);

                            List<NetworkResult> nrs = null;
                             // Create a new computing graph instance
                             using (IComputeGraph computeGraph = CreateComputGraph(deviceIdx, needBack: false))
                            {
                                 // Run forward part
                                 nrs = ForwardOnSingleDevice(computeGraph, spb, decodingOptions, false);
                            }

                            lock (locker)
                            {
                                batchId2Result.Add(deviceIdx, nrs);
                            }

                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, $"Test error at processor '{deviceIdx}'. Exception = '{err.Message}', Call Stack = '{err.StackTrace}'");
                            throw;
                        }
                    });
                }

                if (dataSizePerGPUMod > 0)
                {
                    ISntPairBatch spb = sntPairBatch.GetRange(m_maxDegressOfParallelism * dataSizePerGPU, dataSizePerGPUMod);

                    List<NetworkResult> nrs2 = null;
                    // Create a new computing graph instance
                    using (IComputeGraph computeGraph = CreateComputGraph(0, needBack: false))
                    {
                        // Run forward part
                        nrs2 = ForwardOnSingleDevice(computeGraph, spb, decodingOptions, false);
                    }

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
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}', Call Stack = '{err.StackTrace}'");
                throw;
            }
        }


        public List<NetworkResult> Test<X>(List<List<List<string>>> inputTokensGroups, List<List<List<string>>> outputTokensGroups, DecodingOptions decodingOptions) where X : ISntPairBatch, new()
        {
            X spb = new X();
            spb.CreateBatch(inputTokensGroups, outputTokensGroups);
            var nrs = RunTest(spb, decodingOptions, RunForwardOnSingleDevice);

            return nrs;
        }

        public void Test<X>(string inputTestFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null) where X : ISntPairBatch, new()
        {
            SntBatchStreamReader<X> reader = new SntBatchStreamReader<X>(inputTestFile, batchSize, decodingOptions.MaxSrcSentLength, srcSpmPath);
            SntBatchStreamWriter writer = new SntBatchStreamWriter(outputFile, tgtSpmPath, outputAlignmentFile);
            RunTest<X>(reader, writer, decodingOptions, RunForwardOnSingleDevice);
        }

        public void Test<X>(string inputTestFile, string inputPromptFile, string outputFile, int batchSize, DecodingOptions decodingOptions, string srcSpmPath, string tgtSpmPath, string outputAlignmentFile = null) where X : ISntPairBatch, new()
        {
            SntPairBatchStreamReader<X> reader = new SntPairBatchStreamReader<X>(inputTestFile, inputPromptFile, batchSize, decodingOptions.MaxSrcSentLength, srcSpmPath, tgtSpmPath);
            SntBatchStreamWriter writer = new SntBatchStreamWriter(outputFile, tgtSpmPath, outputAlignmentFile);
            RunTest<X>(reader, writer, decodingOptions, RunForwardOnSingleDevice);
        }


        internal void RunTest<X>(IBatchStreamReader<X> reader, SntBatchStreamWriter writer, DecodingOptions decodingOptions, Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> ForwardOnSingleDevice) where X : ISntPairBatch, new()
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
                            (int idx, ISntPairBatch spb) = reader.GetNextBatch();
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

                            writer.WriteResults(idx, nrs);
                        }
                    }
                    catch (Exception err)
                    {
                        runningGoodSoFar = false;
                        Logger.WriteLine(Logger.Level.err, $"Test error at processor '{deviceIdx}'. Exception = '{err.Message}', Call Stack = '{err.StackTrace}'");
                        throw;
                    }
                });

                writer.Close();
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}', Call Stack = '{err.StackTrace}'");
                throw;
            }
        }


        public void Valid(IParallelCorpus<ISntPairBatch> validCorpus, List<IMetric> metrics, DecodingOptions decodingOptions)
        {
            Dictionary<int, List<IMetric>> taskId2metrics = new Dictionary<int, List<IMetric>>
            {
                { 0, metrics }
            };
            RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, decodingOptions, true);
        }

        public void Valid(IParallelCorpus<ISntPairBatch> validCorpus, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions) => RunValid(validCorpus, RunForwardOnSingleDevice, taskId2metrics, decodingOptions, true);

        /// <summary>
        /// Evaluate the quality of model on valid corpus.
        /// </summary>
        /// <param name="validCorpus">valid corpus to measure the quality of model</param>
        /// <param name="RunNetwork">The network to run on specific device</param>
        /// <param name="metrics">A set of metrics. The first one is the primary metric</param>
        /// <param name="outputToFile">It indicates if valid corpus and results should be dumped to files</param>
        /// <returns>true if we get a better result on primary metric, otherwise, false</returns>
        internal bool RunValid(IParallelCorpus<ISntPairBatch> validCorpus, Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> RunNetwork, Dictionary<int, List<IMetric>> taskId2metrics, DecodingOptions decodingOptions, bool outputToFile = false, string prefixName = "valid")
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

            List<ISntPairBatch> sntPairBatchs = new List<ISntPairBatch>();
            foreach (ISntPairBatch item in validCorpus)
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

        private void RunValidParallel(Func<IComputeGraph, ISntPairBatch, DecodingOptions, bool, List<NetworkResult>> runNetwork, Dictionary<int, List<IMetric>> metrics, DecodingOptions decodingOptions, string taskPrefixName, bool outputToFile, List<ISntPairBatch> sntPairBatchs)
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

                    ISntPairBatch sntPairBatch = sntPairBatchs[deviceIdx];
                    ISntPairBatch sntPairBatchForValid = sntPairBatch.CloneSrcTokens();

                    // Create a new computing graph instance
                    List<NetworkResult> nrs;
                    using (IComputeGraph computeGraph = CreateComputGraph(deviceIdx, needBack: false))
                    {
                        // Run forward part
                        nrs = runNetwork(computeGraph, sntPairBatchForValid, decodingOptions, true);
                    }

                    lock (locker)
                    {
                        string[] newSrcSnts = new string[sntPairBatch.BatchSize];
                        string[] newRefSnts = new string[sntPairBatch.BatchSize];
                        string[] newHypSnts = new string[sntPairBatch.BatchSize];


                        for (int k = 0; k < nrs.Count; k++)
                        {
                            var hypTkns = nrs[k].Output[0];
                            var refTkns = sntPairBatch.GetTgtTokens(k);
                            var srcTkns = sntPairBatch.GetSrcTokens(0);

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
                                        Logger.WriteLine($"Exception = '{err.Message}', Ref = '{string.Join(" ", refTkns[j])}' Hyp = '{string.Join(" ", hypTkns[j])}', TaskId = '{k}'");
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
                            File.AppendAllLines($"{taskPrefixName}_src.txt", newSrcSnts);
                            File.AppendAllLines($"{taskPrefixName}_ref.txt", newRefSnts);
                            File.AppendAllLines($"{taskPrefixName}_hyp.txt", newHypSnts);
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
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Exception: '{err.Message}'");
                    Logger.WriteLine(Logger.Level.err, ConsoleColor.Red, $"Call stack: '{err.StackTrace}'");
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
                // One network wrapper may have multi-names, so we only save one copy of it
                if (setNetworkWrapper.Contains(pair.Value) == false)
                {
                    setNetworkWrapper.Add(pair.Value);
                    pair.Value.Save(m_modelMetaData);
                }
            }
        }
        internal virtual void LoadParameters()
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                Logger.WriteLine($"Loading parameter '{pair.Key}'");
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
                // One network wrapper may have multi-names, so we only save one copy of it
                if (setNetworkWrapper.Add(mpnw))
                {
                    mpnw.Save(model);
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

                Logger.WriteLine($"Loading parameter '{name}'");
                mpnw.Load(model);
            }
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
                result.AddRange(pair.Value.GetNeuralUnitOnDefaultDevice().GetParams());
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
            Logger.WriteLine($"Registering trainable parameters.");
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
                Logger.WriteLine($"Register network '{name}'");
            }

            if (childValue is IMultiProcessorNetworkWrapper[] networksArray)
            {
                int idx = 0;
                foreach (var network in networksArray)
                {
                    string name2 = $"{name}_{idx}";
                    m_name2network.Add(name2, network);
                    Logger.WriteLine($"Register network '{name2}'");

                    idx++;
                }
            }
        }
    }
}
