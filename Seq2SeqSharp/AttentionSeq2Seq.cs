

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

    public class BeamSearchStatus
    {
        public List<int> OutputIds;
        public float Score;

        public List<IWeightMatrix> HTs;
        public List<IWeightMatrix> CTs;

        public BeamSearchStatus()
        {
            OutputIds = new List<int>();
            HTs = new List<IWeightMatrix>();
            CTs = new List<IWeightMatrix>();

            Score = 1.0f;
        }
    }

    public class AttentionSeq2Seq
    {
        public event EventHandler IterationDone;
        public int m_hiddenDim { get; set; }
        public int m_embeddingDim { get; set; }
        public Corpus TrainCorpus { get; set; }
        private int m_encoderLayerDepth;
        private int m_decoderLayerDepth;
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

        private IEncoder[] m_encoder;
        private int m_encoderDefaultDeviceId = 0;

        private AttentionDecoder[] m_decoder;
        private int m_decoderDefaultDeviceId = 0;

        //The feed forward layer after LSTM layers in decoder
        private FeedForwardLayer[] m_decoderFFLayer;
        private int m_DecoderFFLayerDefaultDeviceId = 0;

        // optimization  hyperparameters
        private float m_regc = 1e-10f; // L2 regularization strength
        private float m_startLearningRate = 0.001f;
        private float m_clipvalue = 3.0f; // clip gradients at this value
        private int m_batchSize = 1;
        private int m_parameterUpdateCount = 0;
        private float m_dropoutRatio = 0.1f;
        private string m_modelFilePath;
        private ArchTypeEnums m_archType = ArchTypeEnums.GPU;
        private EncoderTypeEnums m_encoderType = EncoderTypeEnums.Transformer;
        private int[] m_deviceIds;
        private int m_defaultDeviceId = 0;
        private double m_avgCostPerWordInTotalInLastEpoch = 100000.0;
        private int m_multiHeadNum = 8;


        public AttentionSeq2Seq(string modelFilePath, int batchSize, ArchTypeEnums archType, int[] deviceIds)
        {
            m_batchSize = batchSize;
            m_archType = archType;
            m_deviceIds = deviceIds;
            m_modelFilePath = modelFilePath;

            TensorAllocator.InitDevices(archType, deviceIds);
            SetDefaultDeviceIds(deviceIds.Length);

            Logger.WriteLine($"Loading model from '{modelFilePath}'...");

            ModelAttentionMetaData modelMetaData = new ModelAttentionMetaData();
            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream(m_modelFilePath, FileMode.Open, FileAccess.Read);
            modelMetaData = bf.Deserialize(fs) as ModelAttentionMetaData;

            m_clipvalue = modelMetaData.Clipval;
            m_encoderLayerDepth = modelMetaData.EncoderLayerDepth;
            m_decoderLayerDepth = modelMetaData.DecoderLayerDepth;
            m_hiddenDim = modelMetaData.HiddenDim;
            m_startLearningRate = modelMetaData.LearningRate;
            m_embeddingDim = modelMetaData.EmbeddingDim;
            m_multiHeadNum = modelMetaData.MultiHeadNum;
            m_encoderType = modelMetaData.EncoderType;
            m_maxWord = modelMetaData.MaxCharsGen;
            m_regc = modelMetaData.Regc;
            m_dropoutRatio = modelMetaData.DropoutRatio;
            m_srcWordToIndex = modelMetaData.SrcWordToIndex;
            m_srcIndexToWord = modelMetaData.SrcIndexToWord;
            m_tgtWordToIndex = modelMetaData.TgtWordToIndex;
            m_tgtIndexToWord = modelMetaData.TgtIndexToWord;

            CreateEncoderDecoderEmbeddings();

            m_encoder[m_encoderDefaultDeviceId].Load(fs);
            m_decoder[m_decoderDefaultDeviceId].Load(fs);

            m_srcEmbedding[m_srcEmbeddingDefaultDeviceId].Load(fs);
            m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId].Load(fs);

            m_decoderFFLayer[m_DecoderFFLayerDefaultDeviceId].Load(fs);

            fs.Close();
            fs.Dispose();
        }

        public AttentionSeq2Seq(int inputSize, int hiddenSize, int encoderLayerDepth, int decoderLayerDepth, Corpus trainCorpus, string srcVocabFilePath, string tgtVocabFilePath,
            string srcEmbeddingFilePath, string tgtEmbeddingFilePath, string modelFilePath, int batchSize, float dropoutRatio, int multiHeadNum,
            ArchTypeEnums archType, EncoderTypeEnums encoderType, int[] deviceIds)
        {
            TensorAllocator.InitDevices(archType, deviceIds);
            SetDefaultDeviceIds(deviceIds.Length);

            m_dropoutRatio = dropoutRatio;
            m_batchSize = batchSize;
            m_archType = archType;
            m_modelFilePath = modelFilePath;
            m_deviceIds = deviceIds;
            m_multiHeadNum = multiHeadNum;
            m_encoderType = encoderType;

            TrainCorpus = trainCorpus;
            m_encoderLayerDepth = encoderLayerDepth;
            m_decoderLayerDepth = decoderLayerDepth;
            m_embeddingDim = inputSize;
            m_hiddenDim = hiddenSize;

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
            CreateEncoderDecoderEmbeddings();

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
            m_decoderDefaultDeviceId = (i++) % deviceNum;

            m_DecoderFFLayerDefaultDeviceId = (i++) % deviceNum;
        }

        private void InitWeightsFactory()
        {
            m_weightFactory = new IWeightFactory[m_deviceIds.Length];
            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                m_weightFactory[i] = new WeightTensorFactory();
            }

        }

        private (IEncoder[], AttentionDecoder[]) CreateEncoderDecoder()
        {
            Logger.WriteLine($"Creating encoders and decoders...");

            IEncoder[] encoder = new IEncoder[m_deviceIds.Length];
            AttentionDecoder[] decoder = new AttentionDecoder[m_deviceIds.Length];

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                if (m_encoderType == EncoderTypeEnums.BiLSTM)
                {
                    encoder[i] = new BiEncoder(m_batchSize, m_hiddenDim, m_embeddingDim, m_encoderLayerDepth, m_archType, m_deviceIds[i]);
                    decoder[i] = new AttentionDecoder(m_batchSize, m_hiddenDim, m_embeddingDim, m_hiddenDim * 2, m_decoderLayerDepth, m_archType, m_deviceIds[i]);
                }
                else
                {
                    encoder[i] = new TransformerEncoder(m_batchSize, m_multiHeadNum, m_hiddenDim, m_embeddingDim, m_encoderLayerDepth, m_archType, m_deviceIds[i]);
                    decoder[i] = new AttentionDecoder(m_batchSize, m_hiddenDim, m_embeddingDim, m_hiddenDim, m_decoderLayerDepth, m_archType, m_deviceIds[i]);
                }
            }

            return (encoder, decoder);
        }

        private void CreateEncoderDecoderEmbeddings()
        {
            (m_encoder, m_decoder) = CreateEncoderDecoder();

            m_srcEmbedding = new IWeightMatrix[m_deviceIds.Length];
            m_tgtEmbedding = new IWeightMatrix[m_deviceIds.Length];
            m_decoderFFLayer = new FeedForwardLayer[m_deviceIds.Length];

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                Logger.WriteLine($"Initializing weights for device '{m_deviceIds[i]}'");
                m_srcEmbedding[i] = new WeightTensor(m_srcIndexToWord.Count, m_embeddingDim, m_deviceIds[i]);
                m_tgtEmbedding[i] = new WeightTensor(m_tgtIndexToWord.Count + 3, m_embeddingDim, m_deviceIds[i]);

                m_decoderFFLayer[i] = new FeedForwardLayer(m_hiddenDim, m_tgtIndexToWord.Count + 3, m_archType, m_deviceIds[i]);
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
            List<SntPair> sntPairs = new List<SntPair>();

            TensorAllocator.FreeMemoryAllDevices();

            Logger.WriteLine($"Base learning rate is '{learningRate}' at epoch '{ep}'");

            //Clean caches of parameter optmization
            Logger.WriteLine($"Cleaning cache of weights optmiazation.'");
            CleanWeightCache();

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

                    Reset();

                    //Copy weights from weights kept in default device to all other devices
                    SyncWeights();

                    float cost = 0.0f;
                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        IComputeGraph computeGraph = CreateComputGraph(i);

                        //Bi-directional encoding input source sentences
                        IWeightMatrix encodedWeightMatrix = Encode(computeGraph, srcSnts.GetRange(i * m_batchSize, m_batchSize), m_encoder[i], m_srcEmbedding[i]);

                        //Generate output decoder sentences
                        List<List<string>> predictSentence;
                        float lcost = Decode(tgtSnts.GetRange(i * m_batchSize, m_batchSize), computeGraph, encodedWeightMatrix, m_decoder[i], m_decoderFFLayer[i], 
                            m_tgtEmbedding[i], out predictSentence);

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
                    }
                    else
                    {
                        Logger.WriteLine($"Invalid cost value.");
                    }

                    //Optmize parameters
                    float avgAllLR = UpdateParameters(learningRate, TrainCorpus.BatchSize);
                    m_parameterUpdateCount++;

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
                            Update = m_parameterUpdateCount,
                            ProcessedSentencesInTotal = processedLine,
                            ProcessedWordsInTotal = srcWordCnts * 2 + tgtWordCnts,
                            StartDateTime = startDateTime
                        });
                    }


                    //Save model for each 10000 steps
                    if (m_parameterUpdateCount % 1000 == 0 && m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
                    {
                        Save();
                        TensorAllocator.FreeMemoryAllDevices();
                    }

                    sntPairs.Clear();
                }
            }

            Logger.WriteLine($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");
            if (m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
            {
                Save();
            }

            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
        }

        private IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            return new ComputeGraphTensor(m_weightFactory[deviceIdIdx], m_deviceIds[deviceIdIdx], needBack);
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
        private IWeightMatrix Encode(IComputeGraph g, List<List<string>> inputSentences, IEncoder encoder, IWeightMatrix Embedding)
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
                    else
                    {
                        Logger.WriteLine($"'{inputSentence[i]}' is an unknown word.");
                    }
                    var x = g.PeekRow(Embedding, ix_source);
                    forwardInput.Add(x);
                }
            }

            var forwardInputsM = g.ConcatRows(forwardInput);
            return encoder.Encode(forwardInputsM, g);
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
        private float Decode(List<List<string>> outputSentences, IComputeGraph g, IWeightMatrix encodedOutputs, AttentionDecoder decoder, FeedForwardLayer decoderFFLayer, IWeightMatrix Embedding, out List<List<string>> predictSentence)
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
           
                var inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                var eOutput = decoder.Decode(inputsM, attPreProcessResult, g);
                if (m_dropoutRatio > 0.0f)
                {
                    eOutput = g.Dropout(eOutput, m_dropoutRatio);
                }

                var o = decoderFFLayer.Process(eOutput, g);

                //Softmax for output
//                var o = g.MulAdd(eOutput, Whd, bds);
                var probs = g.Softmax(o, false);

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

                //Hacky: Run backward for last feed forward layer and dropout layer in order to save memory usage, since it's not time sequence dependency
                g.RunTopBackward();
                g.RunTopBackward();
                if (m_dropoutRatio > 0.0f)
                {
                    g.RunTopBackward();
                }
            }

            return cost;
        }

        private float UpdateParameters(float learningRate, int batchSize)
        {
            var models = GetParametersFromDefaultDevice();
            return m_solver.UpdateWeights(models, batchSize, learningRate, m_regc, m_clipvalue);
        }
    
        private List<IWeightMatrix> GetParametersFromDeviceAt(int i)
        {
            var model_i = m_encoder[i].GetParams();
            model_i.AddRange(m_decoder[i].GetParams());
            model_i.Add(m_srcEmbedding[i]);
            model_i.Add(m_tgtEmbedding[i]);

            model_i.AddRange(m_decoderFFLayer[i].GetParams());

            return model_i;
        }

        private List<IWeightMatrix> GetParametersFromDefaultDevice()
        {
            var model = m_encoder[m_encoderDefaultDeviceId].GetParams();
            model.AddRange(m_decoder[m_decoderDefaultDeviceId].GetParams());
            model.Add(m_srcEmbedding[m_srcEmbeddingDefaultDeviceId]);
            model.Add(m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId]);

            model.AddRange(m_decoderFFLayer[m_DecoderFFLayerDefaultDeviceId].GetParams());

            return model;
        }

        /// <summary>
        /// Copy weights in default device to all other devices
        /// </summary>
        private void SyncWeights()
        {
            var model = GetParametersFromDefaultDevice();           
            Parallel.For(0, m_deviceIds.Length, i =>
            {
                var model_i = GetParametersFromDeviceAt(i);
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
                var model_i = GetParametersFromDeviceAt(i);
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
            var model = GetParametersFromDefaultDevice();
            Parallel.For(0, m_deviceIds.Length, i =>
            {
                var model_i = GetParametersFromDeviceAt(i);
                for (int j = 0; j < model.Count; j++)
                {
                    if (model[j] != model_i[j])
                    {
                        model[j].AddGradient(model_i[j]);
                    }
                }
            });           
        }


        private void CleanWeightCache()
        {
            var model = GetParametersFromDefaultDevice();
            m_solver.CleanCache(model);
        }

        private void Reset()
        {
            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                m_weightFactory[i].Clear();

                m_encoder[i].Reset(m_weightFactory[i]);
                m_decoder[i].Reset(m_weightFactory[i]);
            }
        }

        public List<List<string>> Predict(List<string> input, int beamSearchSize = 1)
        {
            var biEncoder = m_encoder[m_defaultDeviceId];
            var srcEmbedding = m_srcEmbedding[m_defaultDeviceId];
            var tgtEmbedding = m_tgtEmbedding[m_defaultDeviceId];
            var decoder = m_decoder[m_defaultDeviceId];
            var decoderFFLayer = m_decoderFFLayer[m_defaultDeviceId];

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();

            var g = CreateComputGraph(m_defaultDeviceId, false);
            Reset();

            List<string> inputSeq = new List<string>();
            inputSeq.Add(m_START);
            inputSeq.AddRange(input);
            inputSeq.Add(m_END);
         
            var inputSeqs = new List<List<string>>();
            inputSeqs.Add(inputSeq);
            IWeightMatrix encodedWeightMatrix = Encode(g, inputSeqs, biEncoder, srcEmbedding);

            var attPreProcessResult = decoder.PreProcess(encodedWeightMatrix, g);

            BeamSearchStatus bss = new BeamSearchStatus();
            bss.OutputIds.Add((int)SENTTAGS.START);
            bss.CTs = decoder.GetCTs();
            bss.HTs = decoder.GetHTs();

            bssList.Add(bss);

            List<BeamSearchStatus> newBSSList = new List<BeamSearchStatus>();
            bool finished = false;
            while (finished == false)
            {
                finished = true;
                for (int i = 0; i < bssList.Count; i++)
                {
                    bss = bssList[i];
                    if (bss.OutputIds[bss.OutputIds.Count - 1] == (int)SENTTAGS.END || bss.OutputIds.Count > m_maxWord)
                    {
                        newBSSList.Add(bss);
                    }
                    else
                    {
                        finished = false;
                        var ix_input = bss.OutputIds[bss.OutputIds.Count - 1];
                        decoder.SetCTs(bss.CTs);
                        decoder.SetHTs(bss.HTs);

                        var x = g.PeekRow(tgtEmbedding, ix_input);
                        var eOutput = decoder.Decode(x, attPreProcessResult, g);
                        var o = decoderFFLayer.Process(eOutput, g);

                        var probs = g.Softmax(o, false);

                        var preds = probs.GetTopNMaxWeightIdx(beamSearchSize);

                        for (int j = 0; j < preds.Count; j++)
                        {
                            BeamSearchStatus newBSS = new BeamSearchStatus();
                            newBSS.OutputIds.AddRange(bss.OutputIds);
                            newBSS.OutputIds.Add(preds[j]);

                            newBSS.CTs = decoder.GetCTs();
                            newBSS.HTs = decoder.GetHTs();

                            var score = probs.GetWeightAt(preds[j]);
                            newBSS.Score = bss.Score;
                            newBSS.Score += (float)(-Math.Log(score));

                            //var lengthPenalty = Math.Pow((5.0f + newBSS.OutputIds.Count) / 6, 0.6);
                            //newBSS.Score /= (float)lengthPenalty;

                            newBSSList.Add(newBSS);
                        }
                    }
                }

                bssList = GetTopNBSS(newBSSList, beamSearchSize);
                newBSSList.Clear();
            }
           
            List<List<string>> results = new List<List<string>>();
            for (int i = 0; i < bssList.Count; i++)
            {
                results.Add(PrintString(bssList[i].OutputIds));                
            }

            return results;
        }

        private List<string> PrintString(List<int> idxs)
        {
            List<string> result = new List<string>();
            foreach (var idx in idxs)
            {
                var letter = m_UNK;
                if (m_tgtIndexToWord.ContainsKey(idx))
                {
                    letter = m_tgtIndexToWord[idx];
                }
                result.Add(letter);
            }

            return result;
        }

        private List<BeamSearchStatus> GetTopNBSS(List<BeamSearchStatus> bssList, int topN)
        {
            FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>> q = new FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>>(topN, new ComparableItemComparer<BeamSearchStatus>(false));

            for (int i = 0; i < bssList.Count; i++)
            {
                q.Enqueue(new ComparableItem<BeamSearchStatus>(bssList[i].Score, bssList[i]));
            }

            return q.Select(x => x.Value).ToList();         
        }


        public void Save()
        {
            ModelAttentionMetaData tosave = new ModelAttentionMetaData();
            tosave.Clipval = m_clipvalue;
            tosave.EncoderLayerDepth = m_encoderLayerDepth;
            tosave.DecoderLayerDepth = m_decoderLayerDepth;
            tosave.HiddenDim = m_hiddenDim;
            tosave.LearningRate = m_startLearningRate;
            tosave.EmbeddingDim = m_embeddingDim;
            tosave.MultiHeadNum = m_multiHeadNum;
            tosave.EncoderType = m_encoderType;
            tosave.MaxCharsGen = m_maxWord;
            tosave.Regc = m_regc;
            tosave.DropoutRatio = m_dropoutRatio;
            tosave.SrcWordToIndex = m_srcWordToIndex;
            tosave.SrcIndexToWord = m_srcIndexToWord;

            tosave.TgtWordToIndex = m_tgtWordToIndex;
            tosave.TgtIndexToWord = m_tgtIndexToWord;

            try
            {
                if (File.Exists(m_modelFilePath))
                {
                    File.Copy(m_modelFilePath, $"{m_modelFilePath}.bak", true);
                }

                BinaryFormatter bf = new BinaryFormatter();
                FileStream fs = new FileStream(m_modelFilePath, FileMode.Create, FileAccess.Write);
                bf.Serialize(fs, tosave);

                m_encoder[m_encoderDefaultDeviceId].Save(fs);
                m_decoder[m_decoderDefaultDeviceId].Save(fs);

                m_srcEmbedding[m_srcEmbeddingDefaultDeviceId].Save(fs);
                m_tgtEmbedding[m_tgtEmbeddingDefaultDeviceId].Save(fs);

                m_decoderFFLayer[m_DecoderFFLayerDefaultDeviceId].Save(fs);

                fs.Close();
                fs.Dispose();
            }
            catch (Exception err)
            {
                Logger.WriteLine($"Failed to save model to file. Exception = '{err.Message}'");
            }
        }
    }

    [Serializable]
    public class ModelAttentionMetaData
    {

        public int MaxCharsGen = 100; // max length of generated sentences  
        public int HiddenDim;
        public int EmbeddingDim;

        // optimization  
        public float Regc = 0.000001f; // L2 regularization strength
        public float LearningRate = 0.01f; // learning rate
        public float Clipval = 5.0f; // clip gradients at this value

        public float DropoutRatio { get; set; }

        public int EncoderLayerDepth { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int MultiHeadNum { get; set; }
        public EncoderTypeEnums EncoderType { get; set; }

        public ConcurrentDictionary<string, int> SrcWordToIndex;
        public ConcurrentDictionary<int, string> SrcIndexToWord;

        public ConcurrentDictionary<string, int> TgtWordToIndex;
        public ConcurrentDictionary<int, string> TgtIndexToWord;
    }
}
