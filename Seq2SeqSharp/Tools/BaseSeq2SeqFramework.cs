using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

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
        readonly float m_gradClip = 3.0f; // clip gradients at this value
        readonly IWeightFactory[] m_weightFactory;
        readonly Corpus m_trainCorpus;
        float m_regc = 1e-10f; // L2 regularization strength
        int m_weightsUpdateCount = 0;
        double m_avgCostPerWordInTotalInLastEpoch = 100000.0;
        object locker = new object();
        SortedList<string, IMultiProcessorNetworkWrapper> name2network;

        public BaseSeq2SeqFramework(Corpus trainCorpus, int[] deviceIds, float gradClip, ProcessorTypeEnums processorType, string modelFilePath)
        {
            m_gradClip = gradClip;
            m_trainCorpus = trainCorpus;
            m_deviceIds = deviceIds;
            m_modelFilePath = modelFilePath;
            m_weightFactory = new IWeightFactory[deviceIds.Length];

            for (int i = 0; i < deviceIds.Length; i++)
            {
                m_weightFactory[i] = new WeightTensorFactory();
            }

            TensorAllocator.InitDevices(processorType, m_deviceIds);
        }

        public IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true, bool visNetwork = false)
        {
            // Free memory
            m_weightFactory[deviceIdIdx].Clear();
            // Create computing graph instance and return it
            return new ComputeGraphTensor(m_weightFactory[deviceIdIdx], m_deviceIds[deviceIdIdx], needBack, visNetwork);
        }


        public bool SaveModel(IModelMetaData modelMetaData)
        {
            try
            {
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

        public void TrainOneEpoch(int ep, ILearningRate learningRate, Optimizer solver, IModelMetaData modelMetaData,
            Func<IComputeGraph, List<List<string>>, List<List<string>>, int, float> ForwardOnSingleDevice)
        {
            int processedLineInTotal = 0;
            DateTime startDateTime = DateTime.Now;
            double costInTotal = 0.0;
            long srcWordCnts = 0;
            long tgtWordCnts = 0;
            double avgCostPerWordInTotal = 0.0;

            // Shuffle training corpus
            m_trainCorpus.ShuffleAll(ep == 0);

            TensorAllocator.FreeMemoryAllDevices();

            //Clean caches of parameter optmization
            Logger.WriteLine($"Cleaning cache of weights optmiazation.'");
            CleanGradientCache();

            Logger.WriteLine($"Start to process training corpus.");
            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();

            foreach (var sntPairBatch in m_trainCorpus)
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
                        List<List<string>> srcSnts = new List<List<string>>();
                        List<List<string>> tgtSnts = new List<List<string>>();
                        var sLenInBatch = 0;
                        var tLenInBatch = 0;
                        for (int j = 0; j < sntPairBatch_i.BatchSize; j++)
                        {
                            srcSnts.Add(sntPairBatch_i.SntPairs[j].SrcSnt.ToList());
                            sLenInBatch += sntPairBatch_i.SntPairs[j].SrcSnt.Length;

                            tgtSnts.Add(sntPairBatch_i.SntPairs[j].TgtSnt.ToList());
                            tLenInBatch += sntPairBatch_i.SntPairs[j].TgtSnt.Length;
                        }

                        // Create a new computing graph instance
                        IComputeGraph computeGraph_i = CreateComputGraph(i);
                        // Run forward part
                        float lcost = ForwardOnSingleDevice(computeGraph_i, srcSnts, tgtSnts, i);
                        // Run backward part and compute gradients
                        computeGraph_i.Backward();

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
                    solver.UpdateWeights(models, processedLine, lr, m_regc, m_gradClip);

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

                    //Save model for each 10000 steps
                    if (m_weightsUpdateCount % 1000 == 0 && m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
                    {
                        SaveModel(modelMetaData);
                        TensorAllocator.FreeMemoryAllDevices();
                    }

                    sntPairBatchs.Clear();
                }
            }

            Logger.WriteLine($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");
            if (m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
            {
                SaveModel(modelMetaData);
            }

            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
        }

        internal void SaveParameters(Stream stream)
        {
            RegisterTrainableParameters(this);
            foreach (var pair in name2network)
            {
                pair.Value.Save(stream);
            }
        }

        internal void LoadParameters(Stream stream)
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

        internal void CleanGradientCache()
        {
            RegisterTrainableParameters(this);
            foreach (var pair in name2network)
            {
                pair.Value.ZeroGradientCache();
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
