

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

        public List<IWeightTensor> HTs;
        public List<IWeightTensor> CTs;

        public BeamSearchStatus()
        {
            OutputIds = new List<int>();
            HTs = new List<IWeightTensor>();
            CTs = new List<IWeightTensor>();

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
        private IWeightFactory[] m_weightFactory;
        private ConcurrentDictionary<string, int> m_srcWordToIndex;
        private ConcurrentDictionary<int, string> m_srcIndexToWord;
        private List<string> m_srcVocab = new List<string>();
        private ConcurrentDictionary<string, int> m_tgtWordToIndex;
        private ConcurrentDictionary<int, string> m_tgtIndexToWord;
        private List<string> m_tgtVocab = new List<string>();
        private Optimizer m_solver;

        private IWeightTensor[] m_srcEmbedding;
        private int m_srcEmbeddingDefaultDeviceId = 0;

        private IWeightTensor[] m_tgtEmbedding;
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
        private float m_gradClip = 3.0f; // clip gradients at this value
        private int m_batchSize = 1;
        private int m_parameterUpdateCount = 0;
        private float m_dropoutRatio = 0.0f;
        private string m_modelFilePath;
        private EncoderTypeEnums m_encoderType = EncoderTypeEnums.Transformer;
        private int[] m_deviceIds;
        private int m_defaultDeviceId = 0;
        private double m_avgCostPerWordInTotalInLastEpoch = 100000.0;
        private int m_multiHeadNum = 8;
        private int m_warmupSteps = 8000;

        public AttentionSeq2Seq(string modelFilePath, int batchSize, ArchTypeEnums archType, float dropoutRatio, float gradClip, int[] deviceIds)
        {
            m_batchSize = batchSize;
            m_deviceIds = deviceIds;
            m_modelFilePath = modelFilePath;

            TensorAllocator.InitDevices(archType, deviceIds);
            SetDefaultDeviceIds(deviceIds.Length);

            Logger.WriteLine($"Loading model from '{modelFilePath}'...");

            ModelAttentionMetaData modelMetaData = new ModelAttentionMetaData();
            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream(m_modelFilePath, FileMode.Open, FileAccess.Read);
            modelMetaData = bf.Deserialize(fs) as ModelAttentionMetaData;

            m_gradClip = gradClip;
            m_encoderLayerDepth = modelMetaData.EncoderLayerDepth;
            m_decoderLayerDepth = modelMetaData.DecoderLayerDepth;
            m_hiddenDim = modelMetaData.HiddenDim;

            m_startLearningRate = modelMetaData.LearningRate;
            m_warmupSteps = modelMetaData.WarmupSteps;
            m_parameterUpdateCount = modelMetaData.ParameterUpdatesCount;

            m_embeddingDim = modelMetaData.EmbeddingDim;
            m_multiHeadNum = modelMetaData.MultiHeadNum;
            m_encoderType = modelMetaData.EncoderType;
            m_dropoutRatio = dropoutRatio;
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

        public AttentionSeq2Seq(int embeddingDim, int hiddenDim, int encoderLayerDepth, int decoderLayerDepth, Corpus trainCorpus, string srcVocabFilePath, string tgtVocabFilePath,
            string srcEmbeddingFilePath, string tgtEmbeddingFilePath, string modelFilePath, int batchSize, float dropoutRatio, int multiHeadNum, int warmupSteps, float gradClip,
            ArchTypeEnums archType, EncoderTypeEnums encoderType, int[] deviceIds)
        {
            TensorAllocator.InitDevices(archType, deviceIds);
            SetDefaultDeviceIds(deviceIds.Length);

            m_gradClip = gradClip;
            m_dropoutRatio = dropoutRatio;
            m_batchSize = batchSize;
            m_modelFilePath = modelFilePath;
            m_deviceIds = deviceIds;
            m_multiHeadNum = multiHeadNum;
            m_encoderType = encoderType;
            m_warmupSteps = warmupSteps + 1;

            TrainCorpus = trainCorpus;
            m_encoderLayerDepth = encoderLayerDepth;
            m_decoderLayerDepth = decoderLayerDepth;
            m_embeddingDim = embeddingDim;
            m_hiddenDim = hiddenDim;

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
                    encoder[i] = new BiEncoder("BiLSTMEncoder", m_hiddenDim, m_embeddingDim, m_encoderLayerDepth, m_deviceIds[i]);
                    decoder[i] = new AttentionDecoder("AttnLSTMDecoder", m_hiddenDim, m_embeddingDim, m_hiddenDim * 2, m_decoderLayerDepth, m_deviceIds[i]);
                }
                else
                {
                    encoder[i] = new TransformerEncoder("TransformerEncoder", m_multiHeadNum, m_hiddenDim, m_embeddingDim, m_encoderLayerDepth, m_dropoutRatio, m_deviceIds[i]);
                    decoder[i] = new AttentionDecoder("AttnLSTMDecoder", m_hiddenDim, m_embeddingDim, m_hiddenDim, m_decoderLayerDepth, m_deviceIds[i]);
                }
            }

            return (encoder, decoder);
        }

        private void CreateEncoderDecoderEmbeddings()
        {
            (m_encoder, m_decoder) = CreateEncoderDecoder();

            m_srcEmbedding = new IWeightTensor[m_deviceIds.Length];
            m_tgtEmbedding = new IWeightTensor[m_deviceIds.Length];
            m_decoderFFLayer = new FeedForwardLayer[m_deviceIds.Length];

            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                Logger.WriteLine($"Initializing weights for device '{m_deviceIds[i]}'");
                m_srcEmbedding[i] = new WeightTensor(new long[2] { m_srcIndexToWord.Count, m_embeddingDim }, m_deviceIds[i], normal: true, name: "SrcEmbeddings", isTrainable: true);
                m_tgtEmbedding[i] = new WeightTensor(new long[2] { m_tgtIndexToWord.Count + 3, m_embeddingDim }, m_deviceIds[i], normal: true, name: "TgtEmbeddings", isTrainable: true);

                m_decoderFFLayer[i] = new FeedForwardLayer("FeedForward", m_hiddenDim, m_tgtIndexToWord.Count + 3, m_dropoutRatio, m_deviceIds[i]);
            }

            InitWeightsFactory();
        }


        private void LoadWordEmbedding(string extEmbeddingFilePath, IWeightTensor embeddingMatrix, ConcurrentDictionary<string, int> wordToIndex)
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

            m_srcVocab.Add(Corpus.EOS);
            m_srcVocab.Add(Corpus.BOS);
            m_srcVocab.Add(m_UNK);

            m_srcWordToIndex[Corpus.EOS] = (int)SENTTAGS.END;
            m_srcWordToIndex[Corpus.BOS] = (int)SENTTAGS.START;
            m_srcWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_srcIndexToWord[(int)SENTTAGS.END] = Corpus.EOS;
            m_srcIndexToWord[(int)SENTTAGS.START] = Corpus.BOS;
            m_srcIndexToWord[(int)SENTTAGS.UNK] = m_UNK;



            m_tgtVocab.Add(Corpus.EOS);
            m_tgtVocab.Add(Corpus.BOS);
            m_tgtVocab.Add(m_UNK);

            m_tgtWordToIndex[Corpus.EOS] = (int)SENTTAGS.END;
            m_tgtWordToIndex[Corpus.BOS] = (int)SENTTAGS.START;
            m_tgtWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_tgtIndexToWord[(int)SENTTAGS.END] = Corpus.EOS;
            m_tgtIndexToWord[(int)SENTTAGS.START] = Corpus.BOS;
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

            if (trainCorpus != null)
            {
                foreach (SntPairBatch sntPairBatch in trainCorpus)
                {
                    foreach (SntPair sntPair in sntPairBatch.SntPairs)
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
                }
            }

            m_srcVocab.Add(Corpus.EOS);
            m_srcVocab.Add(Corpus.BOS);
            m_srcVocab.Add(m_UNK);

            m_srcWordToIndex[Corpus.EOS] = (int)SENTTAGS.END;
            m_srcWordToIndex[Corpus.BOS] = (int)SENTTAGS.START;
            m_srcWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_srcIndexToWord[(int)SENTTAGS.END] = Corpus.EOS;
            m_srcIndexToWord[(int)SENTTAGS.START] = Corpus.BOS;
            m_srcIndexToWord[(int)SENTTAGS.UNK] = m_UNK;


            m_tgtVocab.Add(Corpus.EOS);
            m_tgtVocab.Add(Corpus.BOS);
            m_tgtVocab.Add(m_UNK);

            m_tgtWordToIndex[Corpus.EOS] = (int)SENTTAGS.END;
            m_tgtWordToIndex[Corpus.BOS] = (int)SENTTAGS.START;
            m_tgtWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            m_tgtIndexToWord[(int)SENTTAGS.END] = Corpus.EOS;
            m_tgtIndexToWord[(int)SENTTAGS.START] = Corpus.BOS;
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

        public void Train(int trainingEpoch, float startLearningRate)
        {
            Logger.WriteLine("Start to train...");
            m_startLearningRate = startLearningRate;
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
            int processedLineInTotal = 0;
            DateTime startDateTime = DateTime.Now;
            double costInTotal = 0.0;
            long srcWordCnts = 0;
            long tgtWordCnts = 0;
            double avgCostPerWordInTotal = 0.0;

            TensorAllocator.FreeMemoryAllDevices();

            //Clean caches of parameter optmization
            Logger.WriteLine($"Cleaning cache of weights optmiazation.'");
            CleanWeightCache();

            Logger.WriteLine($"Start to process training corpus.");
            List<SntPairBatch> sntPairBatchs = new List<SntPairBatch>();

            foreach (var sntPairBatch in TrainCorpus)
            {
                sntPairBatchs.Add(sntPairBatch);
                if (sntPairBatchs.Count == m_deviceIds.Length)
                {
                    float cost = 0.0f;
                    int tlen = 0;
                    int processedLine = 0;

                    // Copy weights from weights kept in default device to all other devices
                    SyncWeights();

                    Parallel.For(0, m_deviceIds.Length, i =>
                    {
                        SntPairBatch sntPairBatch_i = sntPairBatchs[i];

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

                        try
                        {
                            m_weightFactory[i].Clear();

                            // Reset networks
                            m_encoder[i].Reset(m_weightFactory[i], sntPairBatch_i.BatchSize);
                            m_decoder[i].Reset(m_weightFactory[i], sntPairBatch_i.BatchSize);

                            IComputeGraph computeGraph = CreateComputGraph(i);
                            //Bi-directional encoding input source sentences
                            IWeightTensor encodedWeightMatrix = Encode(computeGraph.CreateSubGraph("Encoder"), srcSnts, m_encoder[i], m_srcEmbedding[i], sntPairBatch_i.BatchSize);
                            //Generate output decoder sentences
                            List<List<string>> predictSentence;
                            float lcost = Decode(tgtSnts, computeGraph.CreateSubGraph("Decoder"), encodedWeightMatrix, m_decoder[i], m_decoderFFLayer[i],
                                m_tgtEmbedding[i], sntPairBatch_i.BatchSize, out predictSentence);

                            lock (locker)
                            {
                                cost += lcost;
                                srcWordCnts += sLenInBatch;
                                tgtWordCnts += tLenInBatch;
                                tlen += tLenInBatch;
                                processedLineInTotal += sntPairBatch_i.BatchSize;
                                processedLine += sntPairBatch_i.BatchSize;
                            }

                            //Calculate gradients
                            computeGraph.Backward();
                        }
                        catch (Exception err)
                        {
                            Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}'");
                            Logger.WriteLine(Logger.Level.err, $"Call stack = '{err.StackTrace}'");
                            Logger.WriteLine(Logger.Level.err, $"SrcLength = '{sntPairBatch_i.SntPairs[0].SrcSnt.Length}', BatchSize = '{sntPairBatch_i.BatchSize}'");

                            throw err;
                        }
                    });

                    //Sum up gradients in all devices, and kept it in default device for parameters optmization
                    SyncGradient();

                    //Optmize parameters
                    m_parameterUpdateCount++;
                    var lr = m_startLearningRate * (float)(Math.Min(Math.Pow(m_parameterUpdateCount, -0.5), Math.Pow(m_warmupSteps, -1.5) * m_parameterUpdateCount) / Math.Pow(m_warmupSteps, -0.5));
                    UpdateParameters(lr, TrainCorpus.BatchSize);

                    //Clear gradient over all devices
                    ClearGradient();

                    costInTotal += cost;
                    avgCostPerWordInTotal = costInTotal / tgtWordCnts;
                    if (IterationDone != null && m_parameterUpdateCount % 100 == 0)
                    {
                        IterationDone(this, new CostEventArg()
                        {
                            LearningRate = lr,
                            CostPerWord = cost / tlen,
                            AvgCostInTotal = avgCostPerWordInTotal,
                            Epoch = ep,
                            Update = m_parameterUpdateCount,
                            ProcessedSentencesInTotal = processedLineInTotal,
                            ProcessedWordsInTotal = srcWordCnts + tgtWordCnts,
                            StartDateTime = startDateTime,
                            BatchSize = processedLine / m_deviceIds.Length
                        });
                    }

                    //Save model for each 10000 steps
                    if (m_parameterUpdateCount % 1000 == 0 && m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
                    {
                        Save();
                        TensorAllocator.FreeMemoryAllDevices();
                    }

                    sntPairBatchs.Clear();
                }               
            }

            Logger.WriteLine($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {avgCostPerWordInTotal.ToString("F6")}, AvgCostInLastEpoch = {m_avgCostPerWordInTotalInLastEpoch.ToString("F6")}");
            if (m_avgCostPerWordInTotalInLastEpoch > avgCostPerWordInTotal)
            {
                Save();
            }

            m_avgCostPerWordInTotalInLastEpoch = avgCostPerWordInTotal;
        }

        private IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true, bool visNetwork = false)
        {
            return new ComputeGraphTensor(m_weightFactory[deviceIdIdx], m_deviceIds[deviceIdIdx], needBack, visNetwork);
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
                    s[i].Add(Corpus.EOS);
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
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> inputSentences, IEncoder encoder, IWeightTensor Embedding, int batchSize)
        {
            PadSentences(inputSentences);
            List<IWeightTensor> forwardOutputs = new List<IWeightTensor>();
            List<IWeightTensor> backwardOutputs = new List<IWeightTensor>();

            int seqLen = inputSentences[0].Count;
            List<IWeightTensor> forwardInput = new List<IWeightTensor>();
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
            return encoder.Encode(forwardInputsM, batchSize, g);
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
        private float Decode(List<List<string>> outputSentences, IComputeGraph g, IWeightTensor encodedOutputs, AttentionDecoder decoder, FeedForwardLayer decoderFFLayer, IWeightTensor Embedding, int batchSize, out List<List<string>> predictSentence)
        {
            predictSentence = null;
            float cost = 0.0f;
            var attPreProcessResult = decoder.PreProcess(encodedOutputs, batchSize, g);

            var originalOutputLengths = PadSentences(outputSentences);
            int seqLen = outputSentences[0].Count;

            int[] ix_inputs = new int[batchSize];
            int[] ix_targets = new int[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = (int)SENTTAGS.START;
            }

            for (int i = 0; i < seqLen + 1; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    List<string> outputSentence = outputSentences[j];

                    ix_targets[j] = (int)SENTTAGS.UNK;
                    if (i >= seqLen)
                    {
                        ix_targets[j] = (int)SENTTAGS.END;
                    }
                    else
                    {
                        if (m_tgtWordToIndex.ContainsKey(outputSentence[i]))
                        {
                            ix_targets[j] = m_tgtWordToIndex[outputSentence[i]];
                        }
                    }

                    var x = g.PeekRow(Embedding, ix_inputs[j]);

                    inputs.Add(x);
                }
           
                var inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                var eOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);
                eOutput = decoderFFLayer.Process(eOutput, batchSize, g);

                //Softmax for output
                using (var probs = g.Softmax(eOutput, runGradients: false, inPlace: true))
                {

                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        using (var probs_k = g.PeekRow(probs, k, runGradients: false))
                        {
                            var score_k = probs_k.GetWeightAt(ix_targets[k]);

                            if (i < originalOutputLengths[k] + 1)
                            {
                                cost += (float)-Math.Log(score_k);
                            }

                            probs_k.SetWeightAt(score_k - 1, ix_targets[k]);
                            ix_inputs[k] = ix_targets[k];
                        }
                    }

                    eOutput.CopyWeightsToGradients(probs);
                }
               

                ////Hacky: Run backward for last feed forward layer and dropout layer in order to save memory usage, since it's not time sequence dependency
                g.RunTopBackward();
                if (m_dropoutRatio > 0.0f)
                {
                    g.RunTopBackward();
                }
            }

            return cost;
        }

        private void UpdateParameters(float learningRate, int batchSize)
        {
            var models = GetParametersFromDefaultDevice();
            m_solver.UpdateWeights(models, batchSize, learningRate, m_regc, m_gradClip);
        }
    
        private List<IWeightTensor> GetParametersFromDeviceAt(int i)
        {
            var model_i = m_encoder[i].GetParams();
            model_i.AddRange(m_decoder[i].GetParams());
            model_i.Add(m_srcEmbedding[i]);
            model_i.Add(m_tgtEmbedding[i]);

            model_i.AddRange(m_decoderFFLayer[i].GetParams());

            return model_i;
        }

        private List<IWeightTensor> GetParametersFromDefaultDevice()
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

        private void Reset(int batchSize)
        {
            for (int i = 0; i < m_deviceIds.Length; i++)
            {
                m_weightFactory[i].Clear();

                m_encoder[i].Reset(m_weightFactory[i], batchSize);
                m_decoder[i].Reset(m_weightFactory[i], batchSize);
            }
        }

        public List<List<string>> Predict(List<string> input, int beamSearchSize = 1, int maxOutputLength = 100, int batchSize = 1)
        {
            var biEncoder = m_encoder[m_defaultDeviceId];
            var srcEmbedding = m_srcEmbedding[m_defaultDeviceId];
            var tgtEmbedding = m_tgtEmbedding[m_defaultDeviceId];
            var decoder = m_decoder[m_defaultDeviceId];
            var decoderFFLayer = m_decoderFFLayer[m_defaultDeviceId];

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();

            var g = CreateComputGraph(m_defaultDeviceId, needBack: false);
            Reset(batchSize);

            List<string> inputSeq = new List<string>();
            inputSeq.Add(Corpus.BOS);
            inputSeq.AddRange(input);
            inputSeq.Add(Corpus.EOS);
         
            var inputSeqs = new List<List<string>>();
            inputSeqs.Add(inputSeq);
            IWeightTensor encodedWeightMatrix = Encode(g.CreateSubGraph("Encoder"), inputSeqs, biEncoder, srcEmbedding, batchSize);

            g = g.CreateSubGraph("Decoder");

            var attPreProcessResult = decoder.PreProcess(encodedWeightMatrix, batchSize, g);

            BeamSearchStatus bss = new BeamSearchStatus();
            bss.OutputIds.Add((int)SENTTAGS.START);
            bss.CTs = decoder.GetCTs();
            bss.HTs = decoder.GetHTs();

            bssList.Add(bss);

            List<BeamSearchStatus> newBSSList = new List<BeamSearchStatus>();
            bool finished = false;
            int outputLength = 0;
            while (finished == false && outputLength < maxOutputLength)
            {
                finished = true;
                for (int i = 0; i < bssList.Count; i++)
                {
                    bss = bssList[i];
                    if (bss.OutputIds[bss.OutputIds.Count - 1] == (int)SENTTAGS.END || bss.OutputIds.Count > maxOutputLength)
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
                        var eOutput = decoder.Decode(x, attPreProcessResult, batchSize, g);
                        var o = decoderFFLayer.Process(eOutput, batchSize, g);

                        var probs = g.Softmax(o);

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

                outputLength++;
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


        public void VisualizeNeuralNetwork(string visNNFilePath)
        {
            var encoder = m_encoder[m_defaultDeviceId];
            var srcEmbedding = m_srcEmbedding[m_defaultDeviceId];
            var tgtEmbedding = m_tgtEmbedding[m_defaultDeviceId];
            var decoder = m_decoder[m_defaultDeviceId];
            var decoderFFLayer = m_decoderFFLayer[m_defaultDeviceId];
            var batchSize = 1;

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();

            var g = CreateComputGraph(m_defaultDeviceId, needBack: false, visNetwork: true);
            Reset(batchSize);

            // Run encoder
            List<string> inputSeq = new List<string>();
            inputSeq.Add(Corpus.BOS);
            inputSeq.Add(Corpus.EOS);

            var inputSeqs = new List<List<string>>();
            inputSeqs.Add(inputSeq);
            IWeightTensor encodedWeightMatrix = Encode(g.CreateSubGraph("Encoder"), inputSeqs, encoder, srcEmbedding, batchSize);

            // Prepare for attention over encoder-decoder
            g = g.CreateSubGraph("Decoder");
            var attPreProcessResult = decoder.PreProcess(encodedWeightMatrix, batchSize, g);

            // Run decoder
            var x = g.PeekRow(tgtEmbedding, (int)SENTTAGS.START);
            var eOutput = decoder.Decode(x, attPreProcessResult, batchSize, g);
            var o = decoderFFLayer.Process(eOutput, batchSize, g);
            var probs = g.Softmax(o);

            g.VisualizeNeuralNetToFile(visNNFilePath);
        }

        public void Save()
        {
            ModelAttentionMetaData tosave = new ModelAttentionMetaData();
            tosave.EncoderLayerDepth = m_encoderLayerDepth;
            tosave.DecoderLayerDepth = m_decoderLayerDepth;
            tosave.HiddenDim = m_hiddenDim;
            tosave.LearningRate = m_startLearningRate;
            tosave.WarmupSteps = m_warmupSteps;
            tosave.ParameterUpdatesCount = m_parameterUpdateCount;
            tosave.EmbeddingDim = m_embeddingDim;
            tosave.MultiHeadNum = m_multiHeadNum;
            tosave.EncoderType = m_encoderType;
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
        public int HiddenDim;
        public int EmbeddingDim;

        // optimization  
        public float LearningRate = 0.01f; // learning rate
        public int WarmupSteps = 1600;
        public int ParameterUpdatesCount = 0;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;

        public ConcurrentDictionary<string, int> SrcWordToIndex;
        public ConcurrentDictionary<int, string> SrcIndexToWord;

        public ConcurrentDictionary<string, int> TgtWordToIndex;
        public ConcurrentDictionary<int, string> TgtIndexToWord;
    }
}
