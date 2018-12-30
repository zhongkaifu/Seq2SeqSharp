

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
        private IWeightFactory m_weightFactory;
        private int m_maxWord = 100;
        private ConcurrentDictionary<string, int> m_srcWordToIndex;
        private ConcurrentDictionary<int, string> m_srcIndexToWord;
        private List<string> m_srcVocab = new List<string>();
        private ConcurrentDictionary<string, int> m_tgtWordToIndex;
        private ConcurrentDictionary<int, string> m_tgtIndexToWord;
        private List<string> m_tgtVocab = new List<string>();
        private Optimizer m_solver;
        private IWeightMatrix m_srcEmbedding;
        private IWeightMatrix m_tgtEmbedding;
        private Encoder m_encoder;
        private Encoder m_reversEncoder;
        private AttentionDecoder m_decoder;
        private int m_batchSize = 1;
        private float m_dropoutRatio = 0.1f;
        private string m_modelFilePath;
        private ArchTypeEnums m_archType = ArchTypeEnums.GPU_CUDA;

        //Output Layer Weights
        private IWeightMatrix m_Whd;
        private IWeightMatrix m_bd;

        // optimization  hyperparameters
        private float m_regc = 0.000001f; // L2 regularization strength
        private float m_startLearningRate = 0.001f;
        private float m_clipvalue = 5.0f; // clip gradients at this value


        public AttentionSeq2Seq(string modelFilePath, int batchSize, ArchTypeEnums archType)
        {
            m_archType = archType;

            Load(modelFilePath);
            InitWeightsFactory();

            SetBatchSize(batchSize);
        }

        public AttentionSeq2Seq(int inputSize, int hiddenSize, int depth, Corpus trainCorpus, string srcVocabFilePath, string tgtVocabFilePath, string srcEmbeddingFilePath, string tgtEmbeddingFilePath,
            bool useDropout, string modelFilePath, int batchSize, float dropoutRatio, ArchTypeEnums archType)
        {
            m_dropoutRatio = dropoutRatio;
            m_batchSize = batchSize;
            m_archType = archType;
            m_modelFilePath = modelFilePath;

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

            //If pre-trained embedding weights are speicifed, loading them from files
            if (String.IsNullOrEmpty(srcEmbeddingFilePath) == false)
            {
                Logger.WriteLine($"Loading ExtEmbedding model from '{srcEmbeddingFilePath}' for source side.");
                LoadWordEmbedding(srcEmbeddingFilePath, m_srcEmbedding, m_srcWordToIndex);
            }

            if (String.IsNullOrEmpty(tgtEmbeddingFilePath) == false)
            {
                Logger.WriteLine($"Loading ExtEmbedding model from '{tgtEmbeddingFilePath}' for target side.");
                LoadWordEmbedding(tgtEmbeddingFilePath, m_tgtEmbedding, m_tgtWordToIndex);
            }
        }

        private void SetBatchSize(int batchSize)
        {
            m_batchSize = batchSize;
            if (m_encoder != null)
            {
                m_encoder.SetBatchSize(m_weightFactory, batchSize);
            }

            if (m_reversEncoder != null)
            {
                m_reversEncoder.SetBatchSize(m_weightFactory, batchSize);
            }

            if (m_decoder != null)
            {
                m_decoder.SetBatchSize(m_weightFactory, batchSize);
            }
        }

        private void InitWeightsFactory()
        {
            if (m_archType == ArchTypeEnums.GPU_CUDA)
            {
                m_weightFactory = new WeightTensorFactory();
            }
            else
            {
                m_weightFactory = new WeightMatrixFactory();
            }

        }

        private void InitWeights()
        {
            Logger.WriteLine($"Initializing weights...");

            if (m_archType == ArchTypeEnums.GPU_CUDA)
            {
                m_Whd = new WeightTensor(HiddenSize, m_tgtIndexToWord.Count + 3, true);
                m_bd = new WeightTensor(1, m_tgtIndexToWord.Count + 3, 0);

                m_srcEmbedding = new WeightTensor(m_srcIndexToWord.Count, WordVectorSize, true);
                m_tgtEmbedding = new WeightTensor(m_tgtIndexToWord.Count + 3, WordVectorSize, true);
            }
            else
            {
                m_Whd = new WeightMatrix(HiddenSize, m_tgtIndexToWord.Count + 3, true);
                m_bd = new WeightMatrix(1, m_tgtIndexToWord.Count + 3, 0);

                m_srcEmbedding = new WeightMatrix(m_srcIndexToWord.Count, WordVectorSize, true);
                m_tgtEmbedding = new WeightMatrix(m_tgtIndexToWord.Count + 3, WordVectorSize, true);
            }

            Logger.WriteLine($"Initializing encoders and decoders...");

            m_encoder = new Encoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType);
            m_reversEncoder = new Encoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType);
            m_decoder = new AttentionDecoder(m_batchSize, HiddenSize, WordVectorSize, Depth, m_archType);

            InitWeightsFactory();
        }

        private void InitWeightsFactory(bool isGPU)
        {
            if (isGPU)
            {
                m_weightFactory = new WeightTensorFactory();
            }
            else
            {
                m_weightFactory = new WeightMatrixFactory();
            }

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

        private void TrainEp(int ep, float learningRate)
        {
            int processedLine = 0;
            DateTime startDateTime = DateTime.Now;

            double costInTotal = 0.0;
            long tgtWordCnts = 0;
            double avgCostPerWordInTotal = 0.0;
            double lastAvgCostPerWordInTotal = 100000.0;
            List<SntPair> sntPairs = new List<SntPair>();

            TensorAllocator.FreeMemory();

            Logger.WriteLine($"Base learning rate is '{learningRate}' at epoch '{ep}'");
            CleanWeightsCash(m_encoder, m_reversEncoder, m_decoder, m_Whd, m_bd, m_srcEmbedding, m_tgtEmbedding);
            foreach (var sntPair in TrainCorpus)
            {
                sntPairs.Add(sntPair);

                if (sntPairs.Count == m_batchSize)
                {
                    IComputeGraph g = CreateComputGraph();

                    Reset(m_weightFactory, m_encoder, m_reversEncoder, m_decoder);

                    List<IWeightMatrix> encoded = new List<IWeightMatrix>();
                    List<List<string>> srcSnts = new List<List<string>>();
                    List<List<string>> tgtSnts = new List<List<string>>();

                    var tlen = 0;
                    for (int j = 0; j < m_batchSize; j++)
                    {
                        List<string> srcSnt = new List<string>();
                        srcSnt.Add(m_START);
                        srcSnt.AddRange(sntPairs[j].SrcSnt);
                        srcSnt.Add(m_END);

                        srcSnts.Add(srcSnt);

                        tgtSnts.Add(sntPairs[j].TgtSnt.ToList());

                        tlen += sntPairs[j].TgtSnt.Length;
                    }
                    tgtWordCnts += tlen;

                    //Bi-directional encoding input source sentences
                    IWeightMatrix encodedWeightMatrix = Encode(g, srcSnts, m_encoder, m_reversEncoder, m_srcEmbedding);

                    //Generate output decoder sentences
                    List<List<string>> predictSentence;
                    float cost = DecodeOutput(tgtSnts, g, encodedWeightMatrix, m_decoder, m_Whd, m_bd, m_tgtEmbedding, out predictSentence);

                    //Calculate gradients
                    g.Backward();

                    if (float.IsInfinity(cost) == false && float.IsNaN(cost) == false)
                    {
                        processedLine += m_batchSize;
                        double costPerWord = (cost / tlen);
                        costInTotal += cost;
                        avgCostPerWordInTotal = costInTotal / tgtWordCnts;
                        lastAvgCostPerWordInTotal = avgCostPerWordInTotal;
                    }
                    else
                    {
                        Logger.WriteLine($"Invalid cost value.");

                    }


                    float avgAllLR = UpdateParameters(learningRate, m_encoder, m_reversEncoder, m_decoder, m_Whd, m_bd, m_srcEmbedding, m_tgtEmbedding);

                    if (IterationDone != null && processedLine % 100 == 0)
                    {

                        IterationDone(this, new CostEventArg()
                        {
                            AvgLearningRate = avgAllLR,
                            CostPerWord = cost / tlen,
                            avgCostInTotal = avgCostPerWordInTotal,
                            Epoch = ep,
                            ProcessedInTotal = processedLine,
                            StartDateTime = startDateTime
                        });
                    }


                    //Save model for each 10000 steps
                    if (processedLine % (m_batchSize * 1000) == 0)
                    {
                        Save();
                        TensorAllocator.FreeMemory();
                    }

                    sntPairs.Clear();
                }
            }

            Logger.WriteLine($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish.");

            Save();
        }

        private IComputeGraph CreateComputGraph(bool needBack = true)
        {
            IComputeGraph g;
            if (m_archType == ArchTypeEnums.CPU_MKL)
            {
                g = new ComputeGraphMKL(m_weightFactory, needBack);
            }
            else if (m_archType == ArchTypeEnums.GPU_CUDA)
            {
                g = new ComputeGraphTensor(m_weightFactory, needBack);
            }
            else
            {
                g = new ComputeGraph(m_weightFactory, needBack);
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


        private IWeightMatrix Encode(IComputeGraph g, List<List<string>> inputSentences, Encoder encoder, Encoder reversEncoder, IWeightMatrix Embedding)
        {
            PadSentences(inputSentences);
            List<IWeightMatrix> forwardOutputs = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardOutputs = new List<IWeightMatrix>();

            int seqLen = inputSentences[0].Count;
            for (int i = 0; i < seqLen; i++)
            {
                List<IWeightMatrix> forwardInput = new List<IWeightMatrix>();
                List<IWeightMatrix> backwardInput = new List<IWeightMatrix>();
                for (int j = 0; j < inputSentences.Count; j++)
                {
                    var inputSentence = inputSentences[j];
                    var reversSentence = inputSentence.ToList();
                    reversSentence.Reverse();


                    int ix_source = (int)SENTTAGS.UNK;
                    if (m_srcWordToIndex.ContainsKey(inputSentence[i]))
                    {
                        ix_source = m_srcWordToIndex[inputSentence[i]];
                    }
                    var x = g.PeekRow(Embedding, ix_source);
                    forwardInput.Add(x);


                    int ix_source2 = (int)SENTTAGS.UNK;
                    if (m_srcWordToIndex.ContainsKey(reversSentence[i]))
                    {
                        ix_source2 = m_srcWordToIndex[reversSentence[i]];
                    }
                    var x2 = g.PeekRow(Embedding, ix_source2);
                    backwardInput.Add(x2);
                }

                var eOutput = encoder.Encode(g.ConcatRows(forwardInput), g);
                forwardOutputs.Add(eOutput);

                var eOutput2 = reversEncoder.Encode(g.ConcatRows(backwardInput), g);
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

        private float DecodeOutput(List<List<string>> OutputSentences, IComputeGraph g, IWeightMatrix encodedOutputs, AttentionDecoder decoder,
                   IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix Embedding, out List<List<string>> predictSentence)
        {
            predictSentence = null;

            float cost = 0.0f;

            decoder.PreProcess(encodedOutputs, g);

            var originalOutputLengths = PadSentences(OutputSentences);
            int seqLen = OutputSentences[0].Count;

            int[] ix_inputs = new int[m_batchSize];
            int[] ix_targets = new int[m_batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = (int)SENTTAGS.START;
            }

            var bds = g.RepeatRows(bd, m_batchSize);

            for (int i = 0; i < seqLen + 1; i++)
            {
                List<IWeightMatrix> inputs = new List<IWeightMatrix>();
                for (int j = 0; j < m_batchSize; j++)
                {
                    List<string> OutputSentence = OutputSentences[j];

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

                var eOutput = decoder.Decode(g.ConcatRows(inputs), g);
                if (m_dropoutRatio > 0.0f)
                {
                    eOutput = g.Dropout(eOutput, m_dropoutRatio);
                }


                var o = g.MulAdd2(eOutput, Whd, bds);
                var probs = g.SoftmaxM(o, false);

                o.ReleaseWeight();

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
            IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix s_Embedding, IWeightMatrix t_Embedding)
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(ReversEncoder.getParams());
            model.Add(s_Embedding);
            model.Add(t_Embedding);
            model.Add(Whd);
            model.Add(bd);
            return m_solver.UpdateWeights(model, m_batchSize, learningRate, m_regc, m_clipvalue);
        }

        private void CleanWeightsCash(Encoder encoder, Encoder ReversEncoder, AttentionDecoder decoder, IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix s_Embedding, IWeightMatrix t_Embedding)
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(ReversEncoder.getParams());
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

            var G2 = CreateComputGraph(false);

            Reset(m_weightFactory, m_encoder, m_reversEncoder, m_decoder);

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

                var x2 = G2.PeekRow(m_srcEmbedding, ix);
                var o = m_encoder.Encode(x2, G2);
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

                var x2 = G2.PeekRow(m_srcEmbedding, ix);
                var o = m_reversEncoder.Encode(x2, G2);
                backwardEncoded.Add(o);

            }

            backwardEncoded.Reverse();
            for (int i = 0; i < inputSeq.Count; i++)
            {
                encoded.Add(G2.ConcatColumns(forwardEncoded[i], backwardEncoded[i]));
            }

            IWeightMatrix encodedWeightMatrix = G2.ConcatRows(encoded);

            m_decoder.PreProcess(encodedWeightMatrix, G2);

            var ix_input = (int)SENTTAGS.START;
            while (true)
            {
                var x = G2.PeekRow(m_tgtEmbedding, ix_input);
                var eOutput = m_decoder.Decode(x, G2);
                var o = G2.MulAdd2(eOutput, this.m_Whd, this.m_bd);

                var probs = G2.SoftmaxWithCrossEntropy(o);

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

                m_bd.Save(fs);
                m_decoder.Save(fs);
                m_encoder.Save(fs);
                m_reversEncoder.Save(fs);
                m_Whd.Save(fs);
                m_srcEmbedding.Save(fs);
                m_tgtEmbedding.Save(fs);

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


            m_bd.Load(fs);
            m_decoder.Load(fs);
            m_encoder.Load(fs);
            m_reversEncoder.Load(fs);
            m_Whd.Load(fs);
            m_srcEmbedding.Load(fs);
            m_tgtEmbedding.Load(fs);


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
