using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using Seq2SeqSharp.Metrics;

namespace Seq2SeqSharp.Tools
{
    /// <summary>
    /// This is a framework for neural network training. It includes many core parts, such as backward propagation, parameters updates, 
    /// memory management, computing graph managment, corpus shuffle & batching, I/O for model, logging & monitoring, checkpoints.
    /// You need to create your network inherited from this class, implmenet forward part only and pass it to TrainOneEpoch method for training
    /// </summary>
    abstract public class BaseSeq2SeqFramework
    {
        public event EventHandler IterationDone;

        readonly int[] m_deviceIds;
        public int[] DeviceIds => m_deviceIds;

        readonly string m_modelFilePath;
        float m_regc = 1e-10f; // L2 regularization strength
        int m_weightsUpdateCount = 0;
        double m_avgCostPerWordInTotalInLastEpoch = 10000.0;
        double m_bestPrimaryScore = 0.0f;
        object locker = new object();
        SortedList<string, IMultiProcessorNetworkWrapper> name2network;

        public BaseSeq2SeqFramework(int[] deviceIds, ProcessorTypeEnums processorType, string modelFilePath)
        {
            m_deviceIds = deviceIds;
            m_modelFilePath = modelFilePath;
            TensorAllocator.InitDevices(processorType, m_deviceIds);
        }

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true, bool visNetwork = false)
        {
            // Create computing graph instance and return it
            return new ComputeGraphTensor(new WeightTensorFactory(), m_deviceIds[deviceIdIdx], needBack, visNetwork);
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
            DateTime lastCheckPointDateTime = DateTime.Now;
            double costInTotal = 0.0;
            long srcWordCnts = 0;
            long tgtWordCnts = 0;
            double avgCostPerWordInTotal = 0.0;

            TensorAllocator.FreeMemoryAllDevices();

            Logger.WriteLine($"Start to process training corpus.");
            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();

            foreach (var sntPairBatch in trainCorpus)
            {
                sntPairBatchs.Add(sntPairBatch);
                if (sntPairBatchs.Count == m_deviceIds.Length)
                {
                    float cost = 0.0f;
                    int tlen = 0;
                    int processedLine = 0;

                    // Copy weights from weights kept in default device to all other devices
                    CopyWeightsFromDefaultDeviceToAllOtherDevices();

                    // Run forward and backward on all available processors
                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        SntPairBatch sntPairBatch_i = sntPairBatchs[i];
                        // Construct sentences for encoding and decoding
                        List<List<string>> srcTkns = new List<List<string>>();
                        List<List<string>> tgtTkns = new List<List<string>>();
                        var sLenInBatch = 0;
                        var tLenInBatch = 0;
                        for (int j = 0; j < sntPairBatch_i.BatchSize; j++)
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
                            tlen += tLenInBatch;
                            processedLineInTotal += sntPairBatch_i.BatchSize;
                            processedLine += sntPairBatch_i.BatchSize;
                        }
                    });

                    //Sum up gradients in all devices, and kept it in default device for parameters optmization
                    SumGradientsToTensorsInDefaultDevice();

                    //Optmize parameters
                    var lr = learningRate.GetCurrentLearningRate();
                    var models = GetParametersFromDefaultDevice();
                    solver.UpdateWeights(models, processedLine, lr, m_regc, m_weightsUpdateCount + 1);

                    //Clear gradient over all devices
                    ZeroGradientOnAllDevices();

                    costInTotal += cost;
                    avgCostPerWordInTotal = costInTotal / tgtWordCnts;
                    m_weightsUpdateCount++;
                    if (IterationDone != null && m_weightsUpdateCount % 100 == 0)
                    {
                        IterationDone(this, new CostEventArg()
                        {
                            LearningRate = lr,
                            CostPerWord = cost / tlen,
                            AvgCostInTotal = avgCostPerWordInTotal,
                            Epoch = ep,
                            Update = m_weightsUpdateCount,
                            ProcessedSentencesInTotal = processedLineInTotal,
                            ProcessedWordsInTotal = srcWordCnts + tgtWordCnts,
                            StartDateTime = startDateTime
                        });
                    }

                    // Evaluate model every hour and save it if we could get a better one.
                    TimeSpan ts = DateTime.Now - lastCheckPointDateTime;
                    if (ts.TotalHours > 1.0)
                    {
                        CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
                        lastCheckPointDateTime = DateTime.Now;
                    }

                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine(Logger.Level.info, ConsoleColor.Green, $"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");

            CreateCheckPoint(validCorpus, metrics, modelMetaData, ForwardOnSingleDevice, avgCostPerWordInTotal);
            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
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
            Logger.WriteLine(Logger.Level.info, ConsoleColor.Gray, $"Start to Evaluate model...");



            List<string> srcSents = new List<string>();
            List<string> refSents = new List<string>();
            List<string> hypSents = new List<string>();

            // Clear inner status of each metrics
            foreach (var metric in metrics)
            {
                metric.ClearStatus();
            }


            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();
            foreach (var item in validCorpus)
            {
                sntPairBatchs.Add(item);
                if (sntPairBatchs.Count == DeviceIds.Length)
                {
                    // Run forward on all available processors
                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        var sntPairBatch = sntPairBatchs[i];

                        // Construct sentences for encoding and decoding
                        List<List<string>> srcTkns = new List<List<string>>();
                        List<List<string>> refTkns = new List<List<string>>();
                        for (int j = 0; j < sntPairBatch.BatchSize; j++)
                        {
                            srcTkns.Add(sntPairBatch.SntPairs[j].SrcSnt.ToList());
                            refTkns.Add(sntPairBatch.SntPairs[j].TgtSnt.ToList());
                        }

                        List<List<string>> hypTkns = new List<List<string>>();

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
                                foreach (var metric in metrics)
                                {
                                    metric.Evaluate(new List<List<string>>() { refTkns[j] }, hypTkns[j]);
                                }
                            }

                            if (outputToFile)
                            {
                                for (int j = 0; j < sntPairBatch.BatchSize; j++)
                                {
                                    srcSents.Add(String.Join(" ", srcTkns[j]));
                                    refSents.Add(String.Join(" ", refTkns[j]));
                                    hypSents.Add(String.Join(" ", hypTkns[j]));
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
                Logger.WriteLine(Logger.Level.info, ConsoleColor.DarkGreen,$"{metric.Name} = {metric.GetScoreStr()}");
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
            foreach (var pair in name2network)
            {
                pair.Value.Save(stream);
            }
        }

        internal virtual void LoadParameters(Stream stream)
        {
            RegisterTrainableParameters(this);
            foreach (var pair in name2network)
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
            foreach (var pair in name2network)
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
            foreach (var pair in name2network)
            {
                pair.Value.SumGradientsToNetworkOnDefaultDevice();
            }
        }

        internal List<IWeightTensor> GetParametersFromDefaultDevice()
        {
            RegisterTrainableParameters(this);
            List<IWeightTensor> result = new List<IWeightTensor>();
            foreach (var pair in name2network)
            {
                result.AddRange(pair.Value.GetNeuralUnitOnDefaultDevice().GetParams());
            }

            return result;
        }

        internal void ZeroGradientOnAllDevices()
        {
            RegisterTrainableParameters(this);
            foreach (var pair in name2network)
            {
                pair.Value.ZeroGradientsOnAllDevices();
            }
        }

        internal void RegisterTrainableParameters(object obj)
        {
            if (name2network != null)
            {
                return;
            }
            Logger.WriteLine($"Registering trainable parameters.");
            name2network = new SortedList<string, IMultiProcessorNetworkWrapper>();

            foreach (FieldInfo childFieldInfo in obj.GetType().GetFields(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                    object childValue = childFieldInfo.GetValue(obj);
                    var name = childFieldInfo.Name;
                    Register(childValue, name);
            }
            foreach (PropertyInfo childPropertyInfo in obj.GetType().GetProperties(BindingFlags.NonPublic | BindingFlags.Instance))
            {
                    object childValue = childPropertyInfo.GetValue(obj);
                    var name = childPropertyInfo.Name;
                    Register(childValue, name);
            }
        }

        private void Register(object childValue, string name)
        {
            var networks = childValue as IMultiProcessorNetworkWrapper;
            if (networks != null)
            {
                name2network.Add(name, networks);
                Logger.WriteLine($"Register network '{name}'");
            }
        }
    }
}
