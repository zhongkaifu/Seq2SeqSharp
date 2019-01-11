

using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.CUDA;

namespace Seq2SeqSharp
{
    public enum SENTTAGS
    {
        END = 0,
        START,
        UNK
    }

    public class AttentionSeq2Seq
    {
        public event EventHandler IterationDone;
        public int HiddenSize { get; set; }
        public int WordVectorSize { get; set; }
        public Corpus TrainCorpus { get; set; }
        public int Depth { get; set; }


        private const string m_UNK = "<UNK>";
        private const string m_END = "<END>";
        private const string m_START = "<START>";
        private IWeightFactory[] m_weightFactory;
        private int m_maxWord = 100;
        private ConcurrentDictionary<string, int> m_srcWordToIndex;
        private ConcurrentDictionary<int, string> m_srcIndexToWord;
        private List<string> m_srcVocab = new List<string>();
        private ConcurrentDictionary<string, int> m_tgtWordToIndex;
        private ConcurrentDictionary<int, string> m_tgtIndexToWord;
        private List<string> m_tgtVocab = new List<string>();
        private Optimizer m_solver;

        private IWeightMatrix[] m_srcEmbedding;
        private int m_srcEmbeddingDefaultDeviceId = 0;

        private IWeightMatrix[] m_tgtEmbedding;
        private int m_tgtEmbeddingDefaultDeviceId = 0;

        private Encoder[] m_encoder;
        private int m_encoderDefaultDeviceId = 0;

        private Encoder[] m_reversEncoder;
        private int m_reversEncoderDefaultDeviceId = 0;

        private AttentionDecoder[] m_decoder;
        private int m_decoderDefaultDeviceId = 0;

        //Output Layer Weights
        private IWeightMatrix[] m_Whd;
        private int m_WhdDefaultDeviceId = 0;

        private IWeightMatrix[] m_bd;
        private int m_bdDefaultDeviceId = 0;


        // optimization  hyperparameters
        private float m_regc = 0.000001f; // L2 regularization strength
        private float m_startLearningRate = 0.001f;
        private float m_clipvalue = 5.0f; // clip gradients at this value
        private int m_batchSize = 1;
        private float m_dropoutRatio = 0.1f;
        private string m_modelFilePath;
        private ArchTypeEnums m_archType = ArchTypeEnums.GPU_CUDA;
        private int[] m_deviceIds;
        private int m_defaultDeviceId = 0;

        public AttentionSeq2Seq(string modelFilePath, int batchSize, ArchTypeEnums archType, int[] deviceIds)
        {
            CheckParameters(batchSize, archType, deviceIds);

            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                TensorAllocator.InitDevices(deviceIds);
                SetDefaultDeviceIds(deviceIds.Length);
            }

            m_archType = archType;
            m_deviceIds = deviceIds;

            Load(modelFilePath);
            InitWeightsFactory();

            SetBatchSize(batchSize);
        }

        public AttentionSeq2Seq(int inputSize, int hiddenSize, int depth, Corpus trainCorpus, string srcVocabFilePath, string tgtVocabFilePath, string srcEmbeddingFilePath, string tgtEmbeddingFilePath,
            bool useDropout, string modelFilePath, int batchSize, float dropoutRatio, ArchTypeEnums archType, int[] deviceIds)
        {
            CheckParameters(batchSize, archType, deviceIds);
            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                TensorAllocator.InitDevices(deviceIds);
                SetDefaultDeviceIds(deviceIds.Length);
            }

            m_dropoutRatio = dropoutRatio;
            m_batchSize = batchSize;
            m_archType = archType;
            m_modelFilePath = modelFilePath;
            m_deviceIds = deviceIds;

            TrainCorpus = trainCorpus;
            Depth = depth;
            WordVectorSize = inputSize;
            HiddenSize = hiddenSize;

            //If vocabulary files are specified, we load them from file, otherwise, we build them from training corpus
            if (String.IsNullOrEmpty(srcVocabFilePath) == false && String.IsNullOrEmpty(tgtVocabFilePath) == false)
            {
                Logger.WriteLine($"Loading vocabulary files from '{srcVocabFilePath}' and '{tgtVocabFilePath}'...");
                LoadVocab(srcVocabFilePath, tgtVocabFilePath);
            }
            else
            {
                Logger.WriteLine("Building vocabulary from training corpus...");
                BuildVocab(trainCorpus);
            }

            //Initializng weights in encoders and decoders
            InitWeights();

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                //If pre-trained embedding weights are speicifed, loading them from files
                if (String.IsNullOrEmpty(srcEmbeddingFilePath) == false)
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{srcEmbeddingFilePath}' for source side.");
                    LoadWordEmbedding(srcEmbeddingFilePath, m_srcEmbedding[i], m_srcWordToIndex);
                }

                if (String.IsNullOrEmpty(tgtEmbeddingFilePath) == false)
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{tgtEmbeddingFilePath}' for target side.");
                    LoadWordEmbedding(tgtEmbeddingFilePath, m_tgtEmbedding[i], m_tgtWordToIndex);
                }
            }
        }

        private void SetDefaultDeviceIds(int deviceNum)
        {
            int i = 0;

            m_srcEmbeddingDefaultDeviceId = (i++) % deviceNum;
            m_tgtEmbeddingDefaultDeviceId = (i++) % deviceNum;

            m_encoderDefaultDeviceId = (i++) % deviceNum;
            m_reversEncoderDefaultDeviceId = (i++) % deviceNum;
            m_decoderDefaultDeviceId = (i++) % deviceNum;

            m_WhdDefaultDeviceId = (i++) % deviceNum;
            m_bdDefaultDeviceId = (i++) % deviceNum;
        }

        private static void CheckParameters(int batchSize, ArchTypeEnums archType, int[] deviceIds)
        {
            if (archType != ArchTypeEnums.GPU_CUDA)
            {
                if (batchSize != 1 || deviceIds.Length != 1)
                {
                    throw new ArgumentException($"Batch size and device Ids length must be 1 if arch type is not GPU");
                }
            }
        }

        private void SetBatchSize(int batchSize)
        {
            m_batchSize = batchSize;

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                if (m_encoder[i] != null)
                {
                    m_encoder[i].SetBatchSize(m_weightFactory[i], batchSize);
                }

                if (m_reversEncoder[i] != null)
                {
                    m_reversEncoder[i].SetBatchSize(m_weightFactory[i], batchSize);
                }

                if (m_decoder[i] != null)
                {
                    m_decoder[i].SetBatchSize(m_weightFactory[i], batchSize);
                }
            }
        }

        private void InitWeightsFactory()
        {
            m_weightFactory = new IWeightFactory[m_deviceIds.Length];
            if (m_archType == ArchTypeEnums.GPU_CUDA)
            {
                for (int i = 0; i < m_deviceIds.Length; i++)
                {
                    m_weightFactory[i] = new WeightTensorFactory();
                }
            }
            else
            {
                for (int i = 0; i < m_deviceIds.Length; i++)
                {
                    m_weightFactory[i] = new WeightMatrixFactory();
                }
            }

        }

        private void InitWeights()
        {
            Logger.WriteLine($"Initializing weights...");

            m_Whd = new IWeightMatrix[m_deviceIds.Length];
            m_bd = new IWeightMatrix[m_deviceIds.Length];
            m_srcEmbedding = new IWeightMatrix[m_deviceIds.Length];
            m_tgtEmbedding = new IWeightMatrix[m_deviceIds.Length];

            m_encoder = new Encoder[m_deviceIds.Length];
            m_reversEncoder = new Encoder[m_deviceIds.Length];
            m_decoder = new AttentionDecoder[m_deviceIds.Length];

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                Logger.WriteLine($"Initializing weights for device '{m_deviceIds[i]}'");
                if (m_archType == ArchTypeEnums.GPU_CUDA)
                {
                    m_Whd[i] = new WeightTensor(HiddenSize, m_tgtIndexToWord.Count + 3, m_deviceIds[i], true);
                    m_bd[i] = new WeightTensor(1, m_tgtIndexToWord.Count + 3, 0, m_deviceIds[i]);

                    m_srcEmbedding[i] = new WeightTensor(m_srcIndexToWord.Count, WordVectorSize, m_deviceIds[i], true);
                    m_tgtEmbedding[i] = new WeightTensor(m_tgtIndexToWord.Count + 3, WordVectorSize, m_deviceIds[i], true);
                }
                else
                {
                    m_Whd[i] = new WeightMatrix(HiddenSize, m_tgtIndexToWord.Count + 3, true);
                    m_bd[i] = new WeightMatrix(1, m_tgtIndexToWord.Count + 3, 0);

                    m_srcEmbedding[i] = new WeightMatrix(m_srcIndexToWord.Count, WordVectorSize, true);
                    m_tgtEmbedding[i] = new WeightMatrix(m_tgtIndexToWord.Count + 3, WordVectorSize, true);
                }

                Logger.WriteLine($"Initializing encoders and decoders for device '{m_deviceIds[i]}'...");

                m_encoder[i] = new Encoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType, m_deviceIds[i]);
                m_reversEncoder[i] = new Encoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType, m_deviceIds[i]);
                m_decoder[i] = new AttentionDecoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType, m_deviceIds[i]);
            }

            InitWeightsFactory();
        }


        private void LoadWordEmbedding(string extEmbeddingFilePath, IWeightMatrix embeddingMatrix, ConcurrentDictionary<string, int> wordToIndex)
        {
            Txt2Vec.Model extEmbeddingModel = new Txt2Vec.Model();
            extEmbeddingModel.LoadBinaryModel(extEmbeddingFilePath);

            if (extEmbeddingModel.VectorSize != embeddingMatrix.Columns)
            {
                throw new ArgumentException($"Inconsistent embedding size. ExtEmbeddingModel size = '{extEmbeddingModel.VectorSize}', EmbeddingMatrix column size = '{embeddingMatrix.Columns}'");
            }

            foreach (KeyValuePair<string, int> pair in wordToIndex)
            {
                float[] vector = extEmbeddingModel.GetVector(pair.Key);

                if (vector != null)
                {
                    
                    embeddingMatrix.SetWeightAtRow(pair.Value, vector);
                }
            }
        }

        /// <summary>
        /// Load vocabulary from given files
        /// </summary>
        /// <param name="srcVocabFilePath"></param>
        /// <param name="tgtVocabFilePath"></param>
        private void LoadVocab(string srcVocabFilePath, string tgtVocabFilePath)
        {
            Logger.WriteLine("Loading vocabulary files...");
            string[] srcVocab = File.ReadAllLines(srcVocabFilePath);
            string[] tgtVocab = File.ReadAllLines(tgtVocabFilePath);

            m_srcWordToIndex = new ConcurrentDictionary<string, int>();
            m_srcIndexToWord = new ConcurrentDictionary<int, string>();
            m_srcVocab = new List<string>();

            m_tgtWordToIndex = new ConcurrentDictionary<string, int>();
            m_tgtIndexToWord = new ConcurrentDictionary<int, string>();
            m_tgtVocab = new List<string>();

            m_srcVocab.Add(m_END);
            m_srcVocab.Add(m_START);
            m_srcVocab.Add(m_UNK);

            m_srcWordToIndex[m_END] = (int)SENTTAGS.END;
            m_srcWordToIndex[m_START] = (int)SENTTAGS.START;
            m_srcWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_srcIndexToWord[(int)SENTTAGS.END] = m_END;
            m_srcIndexToWord[(int)SENTTAGS.START] = m_START;
            m_srcIndexToWord[(int)SENTTAGS.UNK] = m_UNK;



            m_tgtVocab.Add(m_END);
            m_tgtVocab.Add(m_START);
            m_tgtVocab.Add(m_UNK);

            m_tgtWordToIndex[m_END] = (int)SENTTAGS.END;
            m_tgtWordToIndex[m_START] = (int)SENTTAGS.START;
            m_tgtWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_tgtIndexToWord[(int)SENTTAGS.END] = m_END;
            m_tgtIndexToWord[(int)SENTTAGS.START] = m_START;
            m_tgtIndexToWord[(int)SENTTAGS.UNK] = m_UNK;

            //Build word index for both source and target sides
            int q = 3;
            foreach (string line in srcVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                m_srcVocab.Add(word);
                m_srcWordToIndex[word] = q;
                m_srcIndexToWord[q] = word;
                q++;
            }

            q = 3;
            foreach (string line in tgtVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                m_tgtVocab.Add(word);
                m_tgtWordToIndex[word] = q;
                m_tgtIndexToWord[q] = word;
                q++;
            }

        }

        /// <summary>
        /// Build vocabulary from training corpus
        /// </summary>
        /// <param name="trainCorpus"></param>
        /// <param name="minFreq"></param>
        private void BuildVocab(Corpus trainCorpus, int minFreq = 1)
        {
            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            m_srcWordToIndex = new ConcurrentDictionary<string, int>();
            m_srcIndexToWord = new ConcurrentDictionary<int, string>();
            m_srcVocab = new List<string>();

            Dictionary<string, int> t_d = new Dictionary<string, int>();
            m_tgtWordToIndex = new ConcurrentDictionary<string, int>();
            m_tgtIndexToWord = new ConcurrentDictionary<int, string>();
            m_tgtVocab = new List<string>();


            foreach (SntPair sntPair in trainCorpus)
            {
                var item = sntPair.SrcSnt;
                for (int i = 0, n = item.Length; i < n; i++)
                {
                    var txti = item[i];
                    if (s_d.Keys.Contains(txti)) { s_d[txti] += 1; }
                    else { s_d.Add(txti, 1); }
                }

                var item2 = sntPair.TgtSnt;
                for (int i = 0, n = item2.Length; i < n; i++)
                {
                    var txti = item2[i];
                    if (t_d.Keys.Contains(txti)) { t_d[txti] += 1; }
                    else { t_d.Add(txti, 1); }
                }
            }

            m_srcVocab.Add(m_END);
            m_srcVocab.Add(m_START);
            m_srcVocab.Add(m_UNK);

            m_srcWordToIndex[m_END] = (int)SENTTAGS.END;
            m_srcWordToIndex[m_START] = (int)SENTTAGS.START;
            m_srcWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_srcIndexToWord[(int)SENTTAGS.END] = m_END;
            m_srcIndexToWord[(int)SENTTAGS.START] = m_START;
            m_srcIndexToWord[(int)SENTTAGS.UNK] = m_UNK;


            m_tgtVocab.Add(m_END);
            m_tgtVocab.Add(m_START);
            m_tgtVocab.Add(m_UNK);

            m_tgtWordToIndex[m_END] = (int)SENTTAGS.END;
            m_tgtWordToIndex[m_START] = (int)SENTTAGS.START;
            m_tgtWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_tgtIndexToWord[(int)SENTTAGS.END] = m_END;
            m_tgtIndexToWord[(int)SENTTAGS.START] = m_START;
            m_tgtIndexToWord[(int)SENTTAGS.UNK] = m_UNK;


            var q = 3;
            foreach (var ch in s_d)
            {
                if (ch.Value >= minFreq)
                {
                    // add word to vocab
                    m_srcWordToIndex[ch.Key] = q;
                    m_srcIndexToWord[q] = ch.Key;
                    m_srcVocab.Add(ch.Key);
                    q++;
                }

            }

            Logger.WriteLine($"Source language Max term id = '{q}'");


            q = 3;
            foreach (var ch in t_d)
            {
                if (ch.Value >= minFreq)
                {
                    // add word to vocab
                    m_tgtWordToIndex[ch.Key] = q;
                    m_tgtIndexToWord[q] = ch.Key;
                    m_tgtVocab.Add(ch.Key);
                    q++;
                }

            }

            Logger.WriteLine($"Target language Max term id = '{q}'");

        }

        public void Train(int trainingEpoch, float startLearningRate, float gradclip)
        {
            Logger.WriteLine("Start to train...");
            m_startLearningRate = startLearningRate;
            m_clipvalue = gradclip;
            m_solver = new Optimizer();

            float learningRate = m_startLearningRate;
            for (int i = 0; i < trainingEpoch; i++)
            {
                TrainCorpus.ShuffleAll(i == 0);

                TrainEp(i, learningRate);
                learningRate = m_startLearningRate / (1.0f + 0.95f * (i + 1));
            }
        }

        private object locker = new object();

        private void TrainEp(int ep, float learningRate)
        {
            int processedLine = 0;
            DateTime startDateTime = DateTime.Now;

            double costInTotal = 0.0;
            long srcWordCnts = 0;
            long tgtWordCnts = 0;
            double avgCostPerWordInTotal = 0.0;
            double lastAvgCostPerWordInTotal = 100000.0;
            List<SntPair> sntPairs = new List<SntPair>();

            TensorAllocator.FreeMemoryAllDevices();

            Logger.WriteLine($"Base learning rate is '{learningRate}' at epoch '{ep}'");

            //Clean caches of parameter optmization
            Logger.WriteLine($"Cleaning cache of weights optmiazation.'");
            CleanWeightsCash(m_encoder[m_encoderDefaultDeviceId], m_reversEncoder[m_reversEncoderDefaultDeviceId], m_decoder[m_decoderDefaultDeviceId], 
                m_Whd[m_WhdDefaultDeviceId], m_bd[m_bdDefaultDeviceId], m_srcEmbedding[m_srcEmbeddingDefaultDeviceId], m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId]);

            Logger.WriteLine($"Start to process training corpus.");
            foreach (var sntPair in TrainCorpus)
            {
                sntPairs.Add(sntPair);

                if (sntPairs.Count == TrainCorpus.BatchSize)
                {                  
                    List<IWeightMatrix> encoded = new List<IWeightMatrix>();
                    List<List<string>> srcSnts = new List<List<string>>();
                    List<List<string>> tgtSnts = new List<List<string>>();

                    var slen = 0;
                    var tlen = 0;
                    for (int j = 0; j < TrainCorpus.BatchSize; j++)
                    {
                        List<string> srcSnt = new List<string>();

                        //Add BOS and EOS tags to source sentences
                        srcSnt.Add(m_START);
                        srcSnt.AddRange(sntPairs[j].SrcSnt);
                        srcSnt.Add(m_END);

                        srcSnts.Add(srcSnt);
                        tgtSnts.Add(sntPairs[j].TgtSnt.ToList());

                        slen += srcSnt.Count;
                        tlen += sntPairs[j].TgtSnt.Length;
                    }
                    srcWordCnts += slen;
                    tgtWordCnts += tlen;

                    for (int i = 0; i < m_deviceIds.Length; i++)
                    {
                        m_weightFactory[i].Clear();
                        Reset(m_weightFactory[i], m_encoder[i], m_reversEncoder[i], m_decoder[i]);
                    }

                    //Copy weights from weights kept in default device to all other devices
                    SyncWeights();

                    float cost = 0.0f;
                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        IComputeGraph computeGraph = CreateComputGraph(i);

                        //Bi-directional encoding input source sentences
                        IWeightMatrix encodedWeightMatrix = Encode(computeGraph, srcSnts.GetRange(i * m_batchSize, m_batchSize), m_encoder[i], m_reversEncoder[i], m_srcEmbedding[i]);

                        //Generate output decoder sentences
                        List<List<string>> predictSentence;
                        float lcost = Decode(tgtSnts.GetRange(i * m_batchSize, m_batchSize), computeGraph, encodedWeightMatrix, m_decoder[i], m_Whd[i], m_bd[i], m_tgtEmbedding[i], out predictSentence);

                        lock (locker)
                        {
                            cost += lcost;
                        }
                            //Calculate gradients
                            computeGraph.Backward();
                    });

                    //Sum up gradients in all devices, and kept it in default device for parameters optmization
                    SyncGradient();
                   

                    if (float.IsInfinity(cost) == false && float.IsNaN(cost) == false)
                    {
                        processedLine += TrainCorpus.BatchSize;
                        double costPerWord = (cost / tlen);
                        costInTotal += cost;
                        avgCostPerWordInTotal = costInTotal / tgtWordCnts;
                        lastAvgCostPerWordInTotal = avgCostPerWordInTotal;
                    }
                    else
                    {
                        Logger.WriteLine($"Invalid cost value.");
                    }

                    //Optmize parameters
                    float avgAllLR = UpdateParameters(learningRate, m_encoder[m_encoderDefaultDeviceId], m_reversEncoder[m_reversEncoderDefaultDeviceId], m_decoder[m_decoderDefaultDeviceId], 
                        m_Whd[m_WhdDefaultDeviceId], m_bd[m_bdDefaultDeviceId], m_srcEmbedding[m_srcEmbeddingDefaultDeviceId], m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId], TrainCorpus.BatchSize);

                    //Clear gradient over all devices
                    ClearGradient();

                    if (IterationDone != null && processedLine % (100 * TrainCorpus.BatchSize) == 0)
                    {
                        IterationDone(this, new CostEventArg()
                        {
                            AvgLearningRate = avgAllLR,
                            CostPerWord = cost / tlen,
                            avgCostInTotal = avgCostPerWordInTotal,
                            Epoch = ep,
                            ProcessedSentencesInTotal = processedLine,
                            ProcessedWordsInTotal = srcWordCnts * 2 + tgtWordCnts,
                            StartDateTime = startDateTime
                        });
                    }


                    //Save model for each 10000 steps
                    if (processedLine % (TrainCorpus.BatchSize * 1000) == 0)
                    {
                        Save();
                        TensorAllocator.FreeMemoryAllDevices();
                    }

                    sntPairs.Clear();
                }
            }

            Logger.WriteLine($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish.");

            Save();
        }

        private IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            IComputeGraph g;
            if (m_archType == ArchTypeEnums.CPU_MKL)
            {
                g = new ComputeGraphMKL(m_weightFactory[deviceIdIdx], needBack);
            }
            else if (m_archType == ArchTypeEnums.GPU_CUDA)
            {
                g = new ComputeGraphTensor(m_weightFactory[deviceIdIdx], m_deviceIds[deviceIdIdx], needBack);
            }
            else
            {
                g = new ComputeGraph(m_weightFactory[deviceIdIdx], needBack);
            }

            return g;
        }

        private List<int> PadSentences(List<List<string>> s)
        {
            List<int> originalLengths = new List<int>();

            int maxLen = -1;
            foreach (var item in s)
            {
                if (item.Count > maxLen)
                {
                    maxLen = item.Count;
                }

            }

            for (int i = 0; i < s.Count; i++)
            {
                int count = s[i].Count;
                originalLengths.Add(count);

                for (int j = 0; j < maxLen - count; j++)
                {
                    s[i].Add(m_END);
                }
            }

            return originalLengths;
        }

        /// <summary>
        /// Encode source sentences and output encoded weights
        /// </summary>
        /// <param name="g"></param>
        /// <param name="inputSentences"></param>
        /// <param name="encoder"></param>
        /// <param name="reversEncoder"></param>
        /// <param name="Embedding"></param>
        /// <returns></returns>
        private IWeightMatrix Encode(IComputeGraph g, List<List<string>> inputSentences, Encoder encoder, Encoder reversEncoder, IWeightMatrix Embedding)
        {
            PadSentences(inputSentences);
            List<IWeightMatrix> forwardOutputs = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardOutputs = new List<IWeightMatrix>();

            int seqLen = inputSentences[0].Count;
            List<IWeightMatrix> forwardInput = new List<IWeightMatrix>();
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < inputSentences.Count; j++)
                {
                    var inputSentence = inputSentences[j];
                    int ix_source = (int)SENTTAGS.UNK;
                    if (m_srcWordToIndex.ContainsKey(inputSentence[i]))
                    {
                        ix_source = m_srcWordToIndex[inputSentence[i]];
                    }
                    var x = g.PeekRow(Embedding, ix_source);
                    forwardInput.Add(x);
                }
            }

            var forwardInputsM = g.ConcatRows(forwardInput);

            for (int i = 0; i < seqLen; i++)
            {
                var eOutput = encoder.Encode(g.PeekRow(forwardInputsM, i * inputSentences.Count, inputSentences.Count), g);
                forwardOutputs.Add(eOutput);

                var eOutput2 = reversEncoder.Encode(g.PeekRow(forwardInputsM, forwardInputsM.Rows - (i + 1) * inputSentences.Count, inputSentences.Count), g);
                backwardOutputs.Add(eOutput2);
            }

            backwardOutputs.Reverse();

            List<IWeightMatrix> encoded = new List<IWeightMatrix>();
            for (int i = 0; i < seqLen; i++)
            {
                encoded.Add(g.ConcatColumns(forwardOutputs[i], backwardOutputs[i]));
            }

            var encodedOutput = g.ConcatRows(encoded);

            return encodedOutput;
        }

        /// <summary>
        /// Decode output sentences in training
        /// </summary>
        /// <param name="outputSentences"></param>
        /// <param name="g"></param>
        /// <param name="encodedOutputs"></param>
        /// <param name="decoder"></param>
        /// <param name="Whd"></param>
        /// <param name="bd"></param>
        /// <param name="Embedding"></param>
        /// <param name="predictSentence"></param>
        /// <returns></returns>
        private float Decode(List<List<string>> outputSentences, IComputeGraph g, IWeightMatrix encodedOutputs, AttentionDecoder decoder,
                   IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix Embedding, out List<List<string>> predictSentence)
        {
            predictSentence = null;

            float cost = 0.0f;

            var attPreProcessResult = decoder.PreProcess(encodedOutputs, g);

            var originalOutputLengths = PadSentences(outputSentences);
            int seqLen = outputSentences[0].Count;

            int[] ix_inputs = new int[m_batchSize];
            int[] ix_targets = new int[m_batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = (int)SENTTAGS.START;
            }

            var bds = g.RepeatRows(bd, m_batchSize);

            for (int i = 0; i < seqLen + 1; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightMatrix> inputs = new List<IWeightMatrix>();
                for (int j = 0; j < m_batchSize; j++)
                {
                    List<string> OutputSentence = outputSentences[j];

                    ix_targets[j] = (int)SENTTAGS.UNK;
                    if (i >= seqLen)
                    {
                        ix_targets[j] = (int)SENTTAGS.END;
                    }
                    else
                    {
                        if (m_tgtWordToIndex.ContainsKey(OutputSentence[i]))
                        {
                            ix_targets[j] = m_tgtWordToIndex[OutputSentence[i]];
                        }
                    }

                    var x = g.PeekRow(Embedding, ix_inputs[j]);

                    inputs.Add(x);
                }

                //Decode output sentence at position i
                var eOutput = decoder.Decode(g.ConcatRows(inputs), attPreProcessResult, g);
                if (m_dropoutRatio > 0.0f)
                {
                    eOutput = g.Dropout(eOutput, m_dropoutRatio);
                }

                //Softmax for output
                var o = g.MulAdd2(eOutput, Whd, bds);
                var probs = g.SoftmaxM(o, false);

                o.ReleaseWeight();

                //Calculate loss for each word in the batch
                List<IWeightMatrix> probs_g = g.UnFolderRow(probs, m_batchSize, false);
                for (int k = 0; k < m_batchSize; k++)
                {
                    var probs_k = probs_g[k];
                    var score_k = probs_k.GetWeightAt(ix_targets[k]);

                    if (i < originalOutputLengths[k] + 1)
                    {
                        cost += (float)-Math.Log(score_k);
                    }

                    probs_k.SetWeightAt(score_k - 1, ix_targets[k]);

                    ix_inputs[k] = ix_targets[k];
                    probs_k.Dispose();
                }

                o.SetGradientByWeight(probs);
            }

            return cost;
        }

        private float UpdateParameters(float learningRate, Encoder encoder, Encoder ReversEncoder, AttentionDecoder decoder, 
            IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix s_Embedding, IWeightMatrix t_Embedding, int batchSize)
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(ReversEncoder.getParams());
            model.Add(s_Embedding);
            model.Add(t_Embedding);
            model.Add(Whd);
            model.Add(bd);
            return m_solver.UpdateWeights(model, batchSize, learningRate, m_regc, m_clipvalue, m_archType);
        }

        /// <summary>
        /// Copy weights in default device to all other devices
        /// </summary>
        private void SyncWeights()
        {
            var model = m_encoder[m_encoderDefaultDeviceId].getParams();
            model.AddRange(m_decoder[m_decoderDefaultDeviceId].getParams());
            model.AddRange(m_reversEncoder[m_reversEncoderDefaultDeviceId].getParams());
            model.Add(m_srcEmbedding[m_srcEmbeddingDefaultDeviceId]);
            model.Add(m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId]);
            model.Add(m_Whd[m_WhdDefaultDeviceId]);
            model.Add(m_bd[m_bdDefaultDeviceId]);


            Parallel.For(0, m_deviceIds.Length, i =>
            {
                var model_i = m_encoder[i].getParams();
                model_i.AddRange(m_decoder[i].getParams());
                model_i.AddRange(m_reversEncoder[i].getParams());
                model_i.Add(m_srcEmbedding[i]);
                model_i.Add(m_tgtEmbedding[i]);
                model_i.Add(m_Whd[i]);
                model_i.Add(m_bd[i]);

                for (int j = 0; j < model.Count; j++)
                {
                    if (model_i[j] != model[j])
                    {
                        model_i[j].CopyWeights(model[j]);
                    }
                }
            });         
        }

        private void ClearGradient()
        {
            Parallel.For(0, m_deviceIds.Length, i =>
            {
                var model_i = m_encoder[i].getParams();
                model_i.AddRange(m_decoder[i].getParams());
                model_i.AddRange(m_reversEncoder[i].getParams());
                model_i.Add(m_srcEmbedding[i]);
                model_i.Add(m_tgtEmbedding[i]);
                model_i.Add(m_Whd[i]);
                model_i.Add(m_bd[i]);

                for (int j = 0; j < model_i.Count; j++)
                {
                    model_i[j].ClearGradient();
                }
            });
        }

        /// <summary>
        /// Sum up gradients in all devices and keep them in the default device
        /// </summary>
        private void SyncGradient()
        {
            var model = m_encoder[m_encoderDefaultDeviceId].getParams();
            model.AddRange(m_decoder[m_decoderDefaultDeviceId].getParams());
            model.AddRange(m_reversEncoder[m_reversEncoderDefaultDeviceId].getParams());
            model.Add(m_srcEmbedding[m_srcEmbeddingDefaultDeviceId]);
            model.Add(m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId]);
            model.Add(m_Whd[m_WhdDefaultDeviceId]);
            model.Add(m_bd[m_bdDefaultDeviceId]);

            Parallel.For(0, m_deviceIds.Length, i =>
            {
                var model_i = m_encoder[i].getParams();
                model_i.AddRange(m_decoder[i].getParams());
                model_i.AddRange(m_reversEncoder[i].getParams());
                model_i.Add(m_srcEmbedding[i]);
                model_i.Add(m_tgtEmbedding[i]);
                model_i.Add(m_Whd[i]);
                model_i.Add(m_bd[i]);

                for (int j = 0; j < model.Count; j++)
                {
                    if (model[j] != model_i[j])
                    {
                        model[j].AddGradient(model_i[j]);
                    }
                }
            });           
        }


        private void CleanWeightsCash(Encoder encoder, Encoder reversEncoder, AttentionDecoder decoder, IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix s_Embedding, IWeightMatrix t_Embedding)
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(reversEncoder.getParams());
            model.Add(s_Embedding);
            model.Add(t_Embedding);
            model.Add(Whd);
            model.Add(bd);
            m_solver.CleanCash(model);
        }

        private void Reset(IWeightFactory weightFactory, Encoder encoder, Encoder reversEncoder, AttentionDecoder decoder)
        {
            encoder.Reset(weightFactory);
            reversEncoder.Reset(weightFactory);
            decoder.Reset(weightFactory);
        }


        public List<string> Predict(List<string> input)
        {
            List<string> result = new List<string>();

            var g = CreateComputGraph(m_defaultDeviceId, false);
            Reset(m_weightFactory[m_defaultDeviceId], m_encoder[m_defaultDeviceId], m_reversEncoder[m_defaultDeviceId], m_decoder[m_defaultDeviceId]);

            List<string> inputSeq = new List<string>();
            inputSeq.Add(m_START);
            inputSeq.AddRange(input);
            inputSeq.Add(m_END);

            List<string> revseq = inputSeq.ToList();
            revseq.Reverse();

            List<IWeightMatrix> forwardEncoded = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardEncoded = new List<IWeightMatrix>();
            List<IWeightMatrix> encoded = new List<IWeightMatrix>();

            for (int i = 0; i < inputSeq.Count; i++)
            {
                int ix = (int)SENTTAGS.UNK;
                if (m_srcWordToIndex.ContainsKey(inputSeq[i]) == false)
                {
                    Logger.WriteLine($"Unknow input word: {inputSeq[i]}");
                }
                else
                {
                    ix = m_srcWordToIndex[inputSeq[i]];
                }

                var x2 = g.PeekRow(m_srcEmbedding[m_defaultDeviceId], ix);
                var o = m_encoder[m_defaultDeviceId].Encode(x2, g);
                forwardEncoded.Add(o);
            }

            for (int i = 0; i < inputSeq.Count; i++)
            {
                int ix = (int)SENTTAGS.UNK;
                if (m_srcWordToIndex.ContainsKey(revseq[i]) == false)
                {
                    Logger.WriteLine($"Unknow input word: {revseq[i]}");
                }
                else
                {
                    ix = m_srcWordToIndex[revseq[i]];
                }

                var x2 = g.PeekRow(m_srcEmbedding[m_defaultDeviceId], ix);
                var o = m_reversEncoder[m_defaultDeviceId].Encode(x2, g);
                backwardEncoded.Add(o);

            }

            backwardEncoded.Reverse();
            for (int i = 0; i < inputSeq.Count; i++)
            {
                encoded.Add(g.ConcatColumns(forwardEncoded[i], backwardEncoded[i]));
            }

            IWeightMatrix encodedWeightMatrix = g.ConcatRows(encoded);

            var attPreProcessResult = m_decoder[m_defaultDeviceId].PreProcess(encodedWeightMatrix, g);

            var ix_input = (int)SENTTAGS.START;
            while (true)
            {
                var x = g.PeekRow(m_tgtEmbedding[m_defaultDeviceId], ix_input);
                var eOutput = m_decoder[m_defaultDeviceId].Decode(x, attPreProcessResult, g);
                var o = g.MulAdd2(eOutput, m_Whd[m_defaultDeviceId], m_bd[m_defaultDeviceId]);

                var probs = g.SoftmaxWithCrossEntropy(o);

                var pred = probs.GetMaxWeightIdx();
                if (pred == (int)SENTTAGS.END) break; // END token predicted, break out

                if (result.Count > m_maxWord) { break; } // something is wrong 

                var letter2 = m_UNK;
                if (m_tgtIndexToWord.ContainsKey(pred))
                {
                    letter2 = m_tgtIndexToWord[pred];
                }

                result.Add(letter2);
                ix_input = pred;
            }

            return result;
        }

        public void Save()
        {
            ModelAttentionData tosave = new ModelAttentionData();
            tosave.clipval = this.m_clipvalue;
            tosave.Depth = this.Depth;
            tosave.hidden_sizes = this.HiddenSize;
            tosave.learning_rate = m_startLearningRate;
            tosave.letter_size = this.WordVectorSize;
            tosave.max_chars_gen = this.m_maxWord;
            tosave.regc = this.m_regc;
            tosave.DropoutRatio = m_dropoutRatio;
            tosave.s_wordToIndex = m_srcWordToIndex;
            tosave.s_indexToWord = m_srcIndexToWord;

            tosave.t_wordToIndex = m_tgtWordToIndex;
            tosave.t_indexToWord = m_tgtIndexToWord;

            try
            {
                if (File.Exists(m_modelFilePath))
                {
                    File.Copy(m_modelFilePath, $"{m_modelFilePath}.bak", true);
                }

                BinaryFormatter bf = new BinaryFormatter();
                FileStream fs = new FileStream(m_modelFilePath, FileMode.Create, FileAccess.Write);
                bf.Serialize(fs, tosave);

                m_bd[m_bdDefaultDeviceId].Save(fs);
                m_decoder[m_decoderDefaultDeviceId].Save(fs);
                m_encoder[m_encoderDefaultDeviceId].Save(fs);
                m_reversEncoder[m_reversEncoderDefaultDeviceId].Save(fs);
                m_Whd[m_WhdDefaultDeviceId].Save(fs);
                m_srcEmbedding[m_srcEmbeddingDefaultDeviceId].Save(fs);
                m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId].Save(fs);

                fs.Close();
                fs.Dispose();
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to save model to file. Exception = '{err.Message}'");
            }
        }

        public void Load(string modelFilePath)
        {
            Logger.WriteLine($"Loading model from '{modelFilePath}'...");
            m_modelFilePath = modelFilePath;

            ModelAttentionData tosave = new ModelAttentionData();
            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream(m_modelFilePath, FileMode.Open, FileAccess.Read);
            tosave = bf.Deserialize(fs) as ModelAttentionData;

            m_clipvalue = tosave.clipval;
            Depth = tosave.Depth;
            HiddenSize = tosave.hidden_sizes;
            m_startLearningRate = tosave.learning_rate;
            WordVectorSize = tosave.letter_size;
            m_maxWord = 100;
            m_regc = tosave.regc;
            m_dropoutRatio = tosave.DropoutRatio;
            m_srcWordToIndex = tosave.s_wordToIndex;
            m_srcIndexToWord = tosave.s_indexToWord;
            m_tgtWordToIndex = tosave.t_wordToIndex;
            m_tgtIndexToWord = tosave.t_indexToWord;

            InitWeights();

            m_bd[m_bdDefaultDeviceId].Load(fs);
            m_decoder[m_decoderDefaultDeviceId].Load(fs);
            m_encoder[m_encoderDefaultDeviceId].Load(fs);
            m_reversEncoder[m_reversEncoderDefaultDeviceId].Load(fs);
            m_Whd[m_WhdDefaultDeviceId].Load(fs);
            m_srcEmbedding[m_srcEmbeddingDefaultDeviceId].Load(fs);
            m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId].Load(fs);


            fs.Close();
            fs.Dispose();
        }
    }

    [Serializable]
    public class ModelAttentionData
    {

        public int max_chars_gen = 100; // max length of generated sentences  
        public int hidden_sizes;
        public int letter_size;

        // optimization  
        public float regc = 0.000001f; // L2 regularization strength
        public float learning_rate = 0.01f; // learning rate
        public float clipval = 5.0f; // clip gradients at this value


        //public IWeightMatrix s_Wil;
        //public IWeightMatrix t_Wil;
        //public Encoder encoder;
        //public Encoder ReversEncoder;
        //public AttentionDecoder decoder; 
        public float DropoutRatio { get; set; }


        ////Output Layer Weights
        //public IWeightMatrix Whd { get; set; }
        //public IWeightMatrix bd { get; set; }

        public int Depth { get; set; }

        public ConcurrentDictionary<string, int> s_wordToIndex;
        public ConcurrentDictionary<int, string> s_indexToWord;

        public ConcurrentDictionary<string, int> t_wordToIndex;
        public ConcurrentDictionary<int, string> t_indexToWord;
    }
}
