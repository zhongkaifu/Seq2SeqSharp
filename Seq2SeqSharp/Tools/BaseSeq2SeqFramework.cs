using AdvUtils;
using Seq2SeqSharp.Metrics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Tools
{
    /// <summary>
    /// This is a framework for neural network training. It includes many core parts, such as backward propagation, parameters updates, 
    /// memory management, computing graph managment, corpus shuffle & batching, I/O for model, logging & monitoring, checkpoints.
    /// You need to create your network inherited from this class, implmenet forward part only and pass it to TrainOneEpoch method for training
    /// </summary>
    public abstract class BaseSeq2SeqFramework
    {
        public event EventHandler IterationDone;

        private readonly int[] m_deviceIds;
        public int[] DeviceIds => m_deviceIds;

        private readonly string m_modelFilePath;
        private readonly float m_regc = 1e-10f; // L2 regularization strength
        private int m_weightsUpdateCount = 0;
        private double m_avgCostPerWordInTotalInLastEpoch = 10000.0;
        private double m_bestPrimaryScore = 0.0f;
        private readonly object locker = new object();
        private SortedList<string, IMultiProcessorNetworkWrapper> m_name2network;
        DateTime m_lastCheckPointDateTime = DateTime.Now;

        public BaseSeq2SeqFramework(int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath)
        {
            m_deviceIds = deviceIds;
            m_modelFilePath = modelFilePath;
            TensorAllocator.InitDevices(processorType, m_deviceIds);
        }

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            // Create computing graph instance and return it
            return new ComputeGraphTensor(new WeightTensorFactory(), m_deviceIds[deviceIdIdx], needBack);
        }

        public bool SaveModel(IModelMetaData modelMetaData)
        {
            try
            {
                Logger.WriteLine($"Saving model to '{m_modelFilePath}'");

                if (File.Exists(m_modelFilePath))
                {
                    File.Copy(m_modelFilePath, $"{m_modelFilePath}.bak", true);
                }

                BinaryFormatter bf = new BinaryFormatter();
                using (FileStream fs = new FileStream(m_modelFilePath, FileMode.Create, FileAccess.Write))
                {
                    // Save model meta data to the stream
                    bf.Serialize(fs, modelMetaData);
                    // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be saved to given stream
                    SaveParameters(fs);
                }

                return true;
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to save model to file. Exception = '{err.Message}'");
                return false;
            }
        }

        /// <summary>
        /// Load model from given file
        /// </summary>
        /// <param name="InitializeParameters"></param>
        /// <returns></returns>
        public IModelMetaData LoadModel(Func<IModelMetaData, bool> InitializeParameters)
        {
            Logger.WriteLine($"Loading model from '{m_modelFilePath}'...");
            IModelMetaData modelMetaData = null;
            BinaryFormatter bf = new BinaryFormatter();
            using (FileStream fs = new FileStream(m_modelFilePath, FileMode.Open, FileAccess.Read))
            {
                modelMetaData = bf.Deserialize(fs) as IModelMetaData;

                //Initialize parameters on devices
                InitializeParameters(modelMetaData);

                // Load embedding and weights from given model
                // All networks and tensors which are MultiProcessorNetworkWrapper<T> will be loaded from given stream
                LoadParameters(fs);
            }

            return modelMetaData;
        }

        internal void TrainOneEpoch(int ep, ParallelCorpus trainCorpus, ParallelCorpus validCorpus, ILearningRate learningRate, AdamOptimizer solver, List<IMetric> metrics, IModelMetaData modelMetaData,
            Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice)
        {
            int processedLineInTotal = 0;
            DateTime startDateTime = DateTime.Now;
            double costInTotal = 0.0;
            long srcWordCntsInTotal = 0;
            long tgtWordCntsInTotal = 0;
            double avgCostPerWordInTotal = 0.0;

            TensorAllocator.FreeMemoryAllDevices();

            Logger.WriteLine($"Start to process training corpus.");
            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();

            foreach (SntPairBatch sntPairBatch in trainCorpus)
            {
                sntPairBatchs.Add(sntPairBatch);
                if (sntPairBatchs.Count == m_deviceIds.Length)
                {
                    // Copy weights from weights kept in default device to all other devices
                    CopyWeightsFromDefaultDeviceToAllOtherDevices();

                    int batchSplitFactor = 1;
                    bool runNetwordSuccssed = false;

                    while (runNetwordSuccssed == false)
                    {
                        try
                        {
                            (float cost, int sWordCnt, int tWordCnt, int processedLine) = RunNetwork(ForwardOnSingleDevice, sntPairBatchs, batchSplitFactor);
                            processedLineInTotal += processedLine;
                            srcWordCntsInTotal += sWordCnt;
                            tgtWordCntsInTotal += tWordCnt;

                            //Sum up gradients in all devices, and kept it in default device for parameters optmization
                            SumGradientsToTensorsInDefaultDevice();

                            //Optmize parameters
                            float lr = learningRate.GetCurrentLearningRate();
                            List<IWeightTensor> models = GetParametersFromDefaultDevice();
                            solver.UpdateWeights(models, processedLine, lr, m_regc, m_weightsUpdateCount + 1);


                            costInTotal += cost;
                            avgCostPerWordInTotal = costInTotal / tgtWordCntsInTotal;
                            m_weightsUpdateCount++;
                            if (IterationDone != null && m_weightsUpdateCount % 100 == 0)
                            {
                                IterationDone(this, new CostEventArg()
                                {
                                    LearningRate = lr,
                                    CostPerWord = cost / tWordCnt,
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
                        catch (Exception err)
                        {
                            batchSplitFactor *= 2;
                            Logger.WriteLine($"Increase batch split factor to {batchSplitFactor}, and retry it.");

                            if (batchSplitFactor >= sntPairBatchs[0].BatchSize)
                            {
                                Logger.WriteLine($"Batch split factor is larger than batch size, give it up.");
                                throw err;
                            }
                        }
                    }

                    // Evaluate model every hour and save it if we could get a better one.
                    TimeSpan ts = DateTime.Now - m_lastCheckPointDateTime;
                    if (ts.TotalHours > 1.0)
                    {
                        CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
                        m_lastCheckPointDateTime = DateTime.Now;
                    }

                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");

          //  CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
        }

        private (float, int, int, int) RunNetwork(Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice,  List<SntPairBatch> sntPairBatchs, int batchSplitFactor)
        {
            float cost = 0.0f;
            int processedLine = 0;
            int srcWordCnts = 0;
            int tgtWordCnts = 0;

            //Clear gradient over all devices
            ZeroGradientOnAllDevices(); 

            // Run forward and backward on all available processors
            Parallel.For(0, m_deviceIds.Length, i =>
            {
                SntPairBatch sntPairBatch_i = sntPairBatchs[i];
                int batchSegSize = sntPairBatch_i.BatchSize / batchSplitFactor;

                for (int k = 0; k < batchSplitFactor; k++)
                {
                    // Construct sentences for encoding and decoding
                    List<List<string>> srcTkns = new List<List<string>>();
                    List<List<string>> tgtTkns = new List<List<string>>();
                    int sLenInBatch = 0;
                    int tLenInBatch = 0;
                    for (int j = k * batchSegSize; j < (k + 1) * batchSegSize; j++)
                    {
                        srcTkns.Add(sntPairBatch_i.SntPairs[j].SrcSnt.ToList());
                        sLenInBatch += sntPairBatch_i.SntPairs[j].SrcSnt.Length;

                        tgtTkns.Add(sntPairBatch_i.SntPairs[j].TgtSnt.ToList());
                        tLenInBatch += sntPairBatch_i.SntPairs[j].TgtSnt.Length;
                    }

                    float lcost = 0.0f;
                    // Create a new computing graph instance
                    using (IComputeGraph computeGraph_i = CreateComputGraph(i))
                    {
                        // Run forward part
                        lcost = ForwardOnSingleDevice(computeGraph_i, srcTkns, tgtTkns, i, true);
                        // Run backward part and compute gradients
                        computeGraph_i.Backward();
                    }

                    lock (locker)
                    {
                        cost += lcost;
                        srcWordCnts += sLenInBatch;
                        tgtWordCnts += tLenInBatch;
                        processedLine += batchSegSize;
                    }
                }
            });

            return (cost, srcWordCnts, tgtWordCnts, processedLine);
        }

        private void CreateCheckPoint(ParallelCorpus validCorpus, List<IMetric> metrics, IModelMetaData modelMetaData, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice, double avgCostPerWordInTotal)
        {
            if (validCorpus != null)
            {
                // The valid corpus is provided, so evaluate the model.
                if (RunValid(validCorpus, ForwardOnSingleDevice, metrics) == true)
                {
                    SaveModel(modelMetaData);
                }
            }
            else if (m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
            {
                // We don't have valid corpus, so if we could have lower cost, save the model
                SaveModel(modelMetaData);
            }

            TensorAllocator.FreeMemoryAllDevices();
        }

        internal List<List<string>> RunTest(List<List<string>> inputTokens, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> ForwardOnSingleDevice)
        {
            List<List<string>> hypTkns = new List<List<string>>();
            hypTkns.Add(new List<string>());
            hypTkns[0].Add(ParallelCorpus.BOS);

            // Create a new computing graph instance
            using (IComputeGraph computeGraph = CreateComputGraph(DeviceIds[0], needBack: false))
            {
                // Run forward part
                ForwardOnSingleDevice(computeGraph, inputTokens, hypTkns, DeviceIds[0], false);
            }
       
            return hypTkns;
        }

        /// <summary>
        /// Evaluate the quality of model on valid corpus.
        /// </summary>
        /// <param name="validCorpus">valid corpus to measure the quality of model</param>
        /// <param name="RunNetwork">The network to run on specific device</param>
        /// <param name="metrics">A set of metrics. The first one is the primary metric</param>
        /// <param name="outputToFile">It indicates if valid corpus and results should be dumped to files</param>
        /// <returns>true if we get a better result on primary metric, otherwise, false</returns>
        internal bool RunValid(ParallelCorpus validCorpus, Func<IComputeGraph, List<List<string>>, List<List<string>>, int, bool, float> RunNetwork, List<IMetric> metrics, bool outputToFile = false)
        {
            List<string> srcSents = new List<string>();
            List<string> refSents = new List<string>();
            List<string> hypSents = new List<string>();


            // Clear inner status of each metrics
            foreach (IMetric metric in metrics)
            {
                metric.ClearStatus();
            }

            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();
            foreach (SntPairBatch item in validCorpus)
            {
                sntPairBatchs.Add(item);
                if (sntPairBatchs.Count == DeviceIds.Length)
                {

                    // Run forward on all available processors
                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        SntPairBatch sntPairBatch = sntPairBatchs[i];

                        // Construct sentences for encoding and decoding
                        List<List<string>> srcTkns = new List<List<string>>();
                        List<List<string>> refTkns = new List<List<string>>();
                        List<List<string>> hypTkns = new List<List<string>>();
                        for (int j = 0; j < sntPairBatch.BatchSize; j++)
                        {
                            srcTkns.Add(sntPairBatch.SntPairs[j].SrcSnt.ToList());
                            refTkns.Add(sntPairBatch.SntPairs[j].TgtSnt.ToList());
                            hypTkns.Add(new List<string>() { ParallelCorpus.BOS });
                        }

                        // Create a new computing graph instance
                        using (IComputeGraph computeGraph = CreateComputGraph(DeviceIds[i], needBack: false))
                        {
                            // Run forward part
                            RunNetwork(computeGraph, srcTkns, hypTkns, DeviceIds[i], false);
                        }

                        lock (locker)
                        {

                            for (int j = 0; j < hypTkns.Count; j++)
                            {
                                foreach (IMetric metric in metrics)
                                {
                                    metric.Evaluate(new List<List<string>>() { refTkns[j] }, hypTkns[j]);
                                }
                            }

                            if (outputToFile)
                            {
                                for (int j = 0; j < srcTkns.Count; j++)
                                {
                                    srcSents.Add(string.Join(" ", srcTkns[j]));
                                    refSents.Add(string.Join(" ", refTkns[j]));
                                    hypSents.Add(string.Join(" ", hypTkns[j]));
                                }
                            }
                        }


                    });

                    sntPairBatchs.Clear();
                }
            }



            Logger.WriteLine($"Metrics result:");
            foreach (IMetric metric in metrics)
            {
                Logger.WriteLine(Logger.Level.info, ConsoleColor.DarkGreen, $"{metric.Name} = {metric.GetScoreStr()}");
            }

            if (outputToFile)
            {
                File.WriteAllLines("valid_src.txt", srcSents);
                File.WriteAllLines("valid_ref.txt", refSents);
                File.WriteAllLines("valid_hyp.txt", hypSents);
            }

            if (metrics.Count > 0)
            {
                if (metrics[0].GetPrimaryScore() > m_bestPrimaryScore)
                {
                    Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"We got a better score '{metrics[0].GetPrimaryScore().ToString("F")}' on primary metric '{metrics[0].Name}'. The previous score is '{m_bestPrimaryScore.ToString("F")}'");
                    //We have a better primary score on valid set
                    m_bestPrimaryScore = metrics[0].GetPrimaryScore();
                    return true;
                }
            }

            return false;
        }

        internal virtual void SaveParameters(Stream stream)
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                pair.Value.Save(stream);
            }
        }

        internal virtual void LoadParameters(Stream stream)
        {
            RegisterTrainableParameters(this);
            foreach (KeyValuePair<string, IMultiProcessorNetworkWrapper> pair in m_name2network)
            {
                Logger.WriteLine($"Loading parameter '{pair.Key}'");
                pair.Value.Load(stream);
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
            IMultiProcessorNetworkWrapper networks = childValue as IMultiProcessorNetworkWrapper;
            if (networks != null)
            {
                m_name2network.Add(name, networks);
                Logger.WriteLine($"Register network '{name}'");
            }
        }
    }
}
