

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

        public int max_word = 100; // max length of generated sentences 
        public ConcurrentDictionary<string, int> s_wordToIndex; 
        public ConcurrentDictionary<int, string> s_indexToWord;
        public List<string> s_vocab = new List<string>();

        public ConcurrentDictionary<string, int> t_wordToIndex;
        public ConcurrentDictionary<int, string> t_indexToWord;
        public List<string> t_vocab = new List<string>();

        public int HiddenSize { get; set; }
        public int WordVectorSize { get; set; }

        public Corpus TrainCorpus;

        // optimization  hyperparameters
        public float regc = 0.000001f; // L2 regularization strength
        public float StartLearningRate = 0.001f;
        public float clipvalue = 5.0f; // clip gradients at this value


        public Optimizer solver;
        public IWeightMatrix s_Embedding;
        public IWeightMatrix t_Embedding;
        public Encoder encoder;
        public Encoder reversEncoder;
        public AttentionDecoder decoder;

        public bool UseDropout { get; set; } = false;


        //Output Layer Weights
        public IWeightMatrix Whd { get; set; }
        public IWeightMatrix bd { get; set; }

        public int Depth { get; set; }
        public string EncodedModelFilePath { get; set; }

        private const string m_UNK = "<UNK>";
        private const string m_END = "<END>";
        private const string m_START = "<START>";

        public AttentionSeq2Seq()
        {
        }

        public AttentionSeq2Seq(int inputSize, int hiddenSize, int depth, Corpus trainCorpus, string srcVocabFilePath, string tgtVocabFilePath, string srcEmbeddingFilePath, string tgtEmbeddingFilePath,
            bool useSparseFeature, bool useDropout, string modelFilePath)
        {
            this.TrainCorpus = trainCorpus;
            this.Depth=depth;
            // list of sizes of hidden layers
            WordVectorSize = inputSize; // size of word embeddings.
            EncodedModelFilePath = modelFilePath;

            this.HiddenSize = hiddenSize;

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


#if CUDA
            this.Whd = new WeightTensor(HiddenSize , t_vocab.Count + 3, true);
            this.bd = new WeightTensor(1, t_vocab.Count + 3, 0);

            s_Embedding = new WeightTensor(s_vocab.Count, WordVectorSize, true);
            t_Embedding = new WeightTensor(t_vocab.Count + 3, WordVectorSize, true);
#else
            this.Whd = new WeightMatrix(HiddenSize, t_vocab.Count + 3, true);
            this.bd = new WeightMatrix(1, t_vocab.Count + 3, 0);

            s_Embedding = new WeightMatrix(s_vocab.Count, WordVectorSize,   true);
            t_Embedding = new WeightMatrix(t_vocab.Count + 3, WordVectorSize, true);
#endif




            if (String.IsNullOrEmpty(srcEmbeddingFilePath) == false)
            {
                Logger.WriteLine($"Loading ExtEmbedding model from '{srcEmbeddingFilePath}' for source side.");
                LoadWordEmbedding(srcEmbeddingFilePath, s_Embedding, s_wordToIndex);
            }

            if (String.IsNullOrEmpty(tgtEmbeddingFilePath) == false)
            {
                Logger.WriteLine($"Loading ExtEmbedding model from '{tgtEmbeddingFilePath}' for target side.");
                LoadWordEmbedding(tgtEmbeddingFilePath, t_Embedding, t_wordToIndex);
            }

            encoder = new Encoder(HiddenSize, WordVectorSize, depth);
            reversEncoder = new Encoder(HiddenSize, WordVectorSize, depth);

            int sparseFeatureSize = useSparseFeature ? s_vocab.Count : 0;

            decoder = new AttentionDecoder(sparseFeatureSize, HiddenSize, WordVectorSize, depth);
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

            s_wordToIndex = new ConcurrentDictionary<string, int>();
            s_indexToWord = new ConcurrentDictionary<int, string>();
            s_vocab = new List<string>();

            t_wordToIndex = new ConcurrentDictionary<string, int>();
            t_indexToWord = new ConcurrentDictionary<int, string>();
            t_vocab = new List<string>();

            s_vocab.Add(m_END);
            s_vocab.Add(m_START);
            s_vocab.Add(m_UNK);

            s_wordToIndex[m_END] = (int)SENTTAGS.END;
            s_wordToIndex[m_START] = (int)SENTTAGS.START;
            s_wordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            s_indexToWord[(int)SENTTAGS.END] = m_END;
            s_indexToWord[(int)SENTTAGS.START] = m_START;
            s_indexToWord[(int)SENTTAGS.UNK] = m_UNK;

            int q = 3;
            foreach (string line in srcVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                s_vocab.Add(word);
                s_wordToIndex[word] = q;
                s_indexToWord[q] = word;
                q++;
            }

            q = 3;
            foreach (string line in tgtVocab)
            {
                string[] items = line.Split('\t');
                string word = items[0];

                t_vocab.Add(word);
                t_wordToIndex[word] = q;
                t_indexToWord[q] = word;
                q++;
            }

        }


        private void BuildVocab(Corpus trainCorpus, int minFreq = 1)
        {
            // count up all words
            Dictionary<string, int> s_d = new Dictionary<string, int>();
            s_wordToIndex = new ConcurrentDictionary<string, int>();
            s_indexToWord = new ConcurrentDictionary<int, string>();
            s_vocab = new List<string>();

            Dictionary<string, int> t_d = new Dictionary<string, int>();
            t_wordToIndex = new ConcurrentDictionary<string, int>();
            t_indexToWord = new ConcurrentDictionary<int, string>();
            t_vocab = new List<string>();


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

            s_vocab.Add(m_END);
            s_vocab.Add(m_START);
            s_vocab.Add(m_UNK);

            s_wordToIndex[m_END] = (int)SENTTAGS.END;
            s_wordToIndex[m_START] = (int)SENTTAGS.START;
            s_wordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            s_indexToWord[(int)SENTTAGS.END] = m_END;
            s_indexToWord[(int)SENTTAGS.START] = m_START;
            s_indexToWord[(int)SENTTAGS.UNK] = m_UNK;


            // NOTE: start at one because we will have START and END tokens!
            // that is, START token will be index 0 in model word vectors
            // and END token will be index 0 in the next word softmax
            var q = 3;
            foreach (var ch in s_d)
            {
                if (ch.Value >= minFreq)
                {
                    // add word to vocab
                    s_wordToIndex[ch.Key] = q;
                    s_indexToWord[q] = ch.Key;
                    s_vocab.Add(ch.Key);
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
                    t_wordToIndex[ch.Key] = q;
                    t_indexToWord[q] = ch.Key;
                    t_vocab.Add(ch.Key);
                    q++;
                }

            }

            Logger.WriteLine($"Target language Max term id = '{q}'");

        }

        public void Train(int trainingEpoch, float startLearningRate, float gradclip)
        {
            Logger.WriteLine("Start to train...");
            StartLearningRate = startLearningRate;
            clipvalue = gradclip;
            solver = new Optimizer();
         
            for (int i = 0; i < trainingEpoch; i++)
            {
                TrainCorpus.ShuffleAll(i == 0);

                TrainEp(i, StartLearningRate);
                StartLearningRate = StartLearningRate / 2.0f;
            }
        }
        
        private void TrainEp(int ep, float learningRate)
        {
            int processedLine = 0;
            DateTime startDateTime = DateTime.Now;

            double costPerWordInTotal = 0.0;
            double avgCostPerWordInTotal = 0.0;
            double lastAvgCostPerWordInTotal = 100000.0;
            bool updateLR = false;
            int worseCostCnt = 0;

            CleanWeightsCash(encoder, reversEncoder, decoder, Whd, bd, s_Embedding, t_Embedding);

            foreach (SntPair sntPair in TrainCorpus)
            {
                IComputeGraph g;
                float cost;
                List<IWeightMatrix> encoded = new List<IWeightMatrix>();
                SparseWeightMatrix sparseEncoder;

                List<string> srcSnt = new List<string>();
                srcSnt.Add(m_START);
                srcSnt.AddRange(sntPair.SrcSnt);
                srcSnt.Add(m_END);

                g = Encode(srcSnt, out cost, out sparseEncoder, encoded, encoder, reversEncoder, s_Embedding);

                IWeightMatrix encodedWeightMatrix = g.ConcatRows(encoded);

                cost = DecodeOutput(sntPair.TgtSnt, g, cost, sparseEncoder, encodedWeightMatrix, decoder, Whd, bd, t_Embedding);
                g.Backward();

                if (float.IsInfinity(cost) == false && float.IsNaN(cost) == false)
                {
                    processedLine++;
                    double costPerWord = (cost / sntPair.TgtSnt.Length);
                    costPerWordInTotal += costPerWord;
                    avgCostPerWordInTotal = costPerWordInTotal / processedLine;

                    if (avgCostPerWordInTotal > lastAvgCostPerWordInTotal)
                    {
                        worseCostCnt++;

                        if (worseCostCnt > 5)
                        {
                            updateLR = true;
                        }
                    }
                    else
                    {
                        worseCostCnt = 0;
                        updateLR = false;
                    }


                    lastAvgCostPerWordInTotal = avgCostPerWordInTotal;
                }
                else
                {
                    updateLR = true;

                }

                float avgAllLR = UpdateParameters(learningRate, encoder, reversEncoder, decoder, Whd, bd, s_Embedding, t_Embedding, updateLR);
                if (IterationDone != null && processedLine % 100 == 0)
                {

                    IterationDone(this, new CostEventArg()
                    {
                        AvgLearningRate = avgAllLR,
                        CostPerWord = cost / sntPair.TgtSnt.Length,
                        avgCostInTotal = avgCostPerWordInTotal,
                        Epoch = ep,
                        ProcessedInTotal = processedLine,
                        SentenceLength = sntPair.TgtSnt.Length,
                        StartDateTime = startDateTime                      
                    });
                }

                Reset(encoder, reversEncoder, decoder);

                //Save model for each 1000 steps
                if (processedLine % 1000 == 0)
                {
                    Save();
                }
            }



            Save();
        }

        private IComputeGraph Encode(List<string> inputSentence, out float cost, out SparseWeightMatrix sparseWeightMartix, List<IWeightMatrix> encoded, Encoder encoder, Encoder reversEncoder, IWeightMatrix Embedding)
        {
            var reversSentence = inputSentence.ToList();
            reversSentence.Reverse();

#if MKL
            IComputeGraph g = new ComputeGraphMKL();
#elif CUDA
            IComputeGraph g = new ComputeGraphTensor();
#else
            IComputeGraph g = new ComputeGraph();
#endif


            cost = 0.0f;
            SparseWeightMatrix tmpSWM = new SparseWeightMatrix(1, Embedding.Columns);
            List<IWeightMatrix> forwardOutputs = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardOutputs = new List<IWeightMatrix>();


            for (int i = 0; i < inputSentence.Count; i++)
            {
                int ix_source = (int)SENTTAGS.UNK;

                if (s_wordToIndex.ContainsKey(inputSentence[i]))
                {
                    ix_source = s_wordToIndex[inputSentence[i]];
                }
                var x = g.PeekRow(Embedding, ix_source);
                var eOutput = encoder.Encode(x, g);
                forwardOutputs.Add(eOutput);

                tmpSWM.AddWeight(0, ix_source, 1.0f);
            }


            for (int i = 0; i < inputSentence.Count; i++)
            {
                int ix_source2 = (int)SENTTAGS.UNK;

                if (s_wordToIndex.ContainsKey(reversSentence[i]))
                {
                    ix_source2 = s_wordToIndex[reversSentence[i]];
                }

                var x2 = g.PeekRow(Embedding, ix_source2);
                var eOutput2 = reversEncoder.Encode(x2, g);
                backwardOutputs.Add(eOutput2);
            }

            backwardOutputs.Reverse();

            for (int i = 0; i < inputSentence.Count; i++)
            {
                encoded.Add(g.ConcatColumns(forwardOutputs[i], backwardOutputs[i]));
            }

            sparseWeightMartix = tmpSWM;

            return g;
        }

        private float DecodeOutput(string[] OutputSentence, IComputeGraph g, float cost, SparseWeightMatrix sparseInput, IWeightMatrix encodedOutputs, AttentionDecoder decoder, IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix Embedding)
        {
            decoder.PreProcess(encodedOutputs, g);

            int ix_input = (int)SENTTAGS.START;
            for (int i = 0; i < OutputSentence.Length + 1; i++)
            {
                int ix_target = (int)SENTTAGS.UNK;
                if (i == OutputSentence.Length) 
                {
                    ix_target = (int)SENTTAGS.END; 
                }
                else
                {
                    if (t_wordToIndex.ContainsKey(OutputSentence[i]))
                    {
                        ix_target = t_wordToIndex[OutputSentence[i]];
                    }
                }


                var x = g.PeekRow(Embedding, ix_input);
                var eOutput = decoder.Decode(sparseInput, x, encodedOutputs, g);
                if (UseDropout)
                {
                    eOutput = g.Dropout(eOutput, 0.2f);

                }

                var o = g.MulAdd(eOutput, Whd, bd);

                var probs = g.SoftmaxWithCrossEntropy(o);
                var score = probs.GetWeightAt(ix_target);              
                cost += (float)-Math.Log(score);

                probs.SetWeightAt(score - 1, ix_target);
                o.SetGradientByWeight(probs);


                ix_input = ix_target;
            }
            return cost;
        }

        private float UpdateParameters(float learningRate, Encoder encoder, Encoder ReversEncoder, AttentionDecoder decoder, IWeightMatrix Whd, IWeightMatrix bd, IWeightMatrix s_Embedding, IWeightMatrix t_Embedding, bool updateLR = true)
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(ReversEncoder.getParams());
            model.Add(s_Embedding);
            model.Add(t_Embedding);
            model.Add(Whd);
            model.Add(bd);
            return solver.UpdateWeights(model, learningRate, regc, clipvalue, updateLR);
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
            solver.CleanCash(model);
        }

        private void Reset(Encoder encoder, Encoder reversEncoder, AttentionDecoder decoder)
        {
            encoder.Reset();
            reversEncoder.Reset();
            decoder.Reset();
        }


        public List<string> Predict(List<string> input)
        {
            reversEncoder.Reset();
            encoder.Reset();
            decoder.Reset();

            List<string> result = new List<string>();

#if MKL
            var G2 = new ComputeGraphMKL(false);
#elif CUDA
            var G2 = new ComputeGraphTensor(false);
#else
            var G2 = new ComputeGraph(false);
#endif

            List<string> inputSeq = new List<string>();
            inputSeq.Add(m_START);
            inputSeq.AddRange(input);
            inputSeq.Add(m_END);

            List<string> revseq = inputSeq.ToList();
            revseq.Reverse();

            List<IWeightMatrix> forwardEncoded = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardEncoded = new List<IWeightMatrix>();
            List<IWeightMatrix> encoded = new List<IWeightMatrix>();
            SparseWeightMatrix sparseInput = new SparseWeightMatrix(1, s_Embedding.Columns);


            for (int i = 0; i < inputSeq.Count; i++)
            {
                int ix = (int)SENTTAGS.UNK;
                if (s_wordToIndex.ContainsKey(inputSeq[i]) == false)
                {
                    Logger.WriteLine($"Unknow input word: {inputSeq[i]}");
                }
                else
                {
                    ix = s_wordToIndex[inputSeq[i]];
                }

                var x2 = G2.PeekRow(s_Embedding, ix);
                var o = encoder.Encode(x2, G2);
                forwardEncoded.Add(o);

                sparseInput.AddWeight(0, ix, 1.0f);
            }

            for (int i = 0; i < inputSeq.Count; i++)
            {
                int ix = (int)SENTTAGS.UNK;
                if (s_wordToIndex.ContainsKey(revseq[i]) == false)
                {
                    Logger.WriteLine($"Unknow input word: {revseq[i]}");
                }
                else
                {
                    ix = s_wordToIndex[revseq[i]];
                }

                var x2 = G2.PeekRow(s_Embedding, ix);
                var o = reversEncoder.Encode(x2, G2);
                backwardEncoded.Add(o);

            }

            backwardEncoded.Reverse();
            for (int i = 0; i < inputSeq.Count; i++)
            {
                encoded.Add(G2.ConcatColumns(forwardEncoded[i], backwardEncoded[i]));
              //  encoded.Add(G2.Add(forwardEncoded[i], backwardEncoded[i]));
            }

            IWeightMatrix encodedWeightMatrix = G2.ConcatRows(encoded);

            decoder.PreProcess(encodedWeightMatrix, G2);

            var ix_input = (int)SENTTAGS.START;
            while (true)
            {
                var x = G2.PeekRow(t_Embedding, ix_input);
                var eOutput = decoder.Decode(sparseInput, x, encodedWeightMatrix, G2);
                if (UseDropout)
                {
                    G2.DropoutPredict(eOutput, 0.2f);
                }

                var o = G2.MulAdd(eOutput, this.Whd, this.bd);

                var probs = G2.SoftmaxWithCrossEntropy(o);
                var weight = probs.ToWeightArray();

                var maxv = weight[0];
                var maxi = 0;
                for (int i = 1; i < weight.Length; i++)
                {
                    if (weight[i] > maxv)
                    {
                        maxv = weight[i];
                        maxi = i;
                    }
                }
                var pred = maxi;

                if (pred == (int)SENTTAGS.END) break; // END token predicted, break out

                if (result.Count > max_word) { break; } // something is wrong 

                var letter2 = m_UNK;
                if (t_indexToWord.ContainsKey(pred))
                {
                    letter2 = t_indexToWord[pred];
                }

                result.Add(letter2);
                ix_input = pred;
            }

            return result;
        }

        public void Save()
        {
            ModelAttentionData tosave = new ModelAttentionData();
            tosave.bd = this.bd;
            tosave.clipval = this.clipvalue;
            tosave.decoder = this.decoder;
            tosave.Depth = this.Depth;
            tosave.encoder = this.encoder;
            tosave.hidden_sizes = this.HiddenSize;
            tosave.learning_rate = StartLearningRate;
            tosave.letter_size = this.WordVectorSize;
            tosave.max_chars_gen = this.max_word;
            tosave.regc = this.regc;
            tosave.ReversEncoder = this.reversEncoder;
            tosave.UseDropout = this.UseDropout;
            tosave.Whd = this.Whd;
            tosave.s_Wil = this.s_Embedding;
            tosave.s_wordToIndex = s_wordToIndex;
            tosave.s_indexToWord = s_indexToWord;

            tosave.t_Wil = this.t_Embedding;
            tosave.t_wordToIndex = t_wordToIndex;
            tosave.t_indexToWord = t_indexToWord;

            try
            {
                if (File.Exists(EncodedModelFilePath))
                {
                    File.Copy(EncodedModelFilePath, $"{EncodedModelFilePath}.bak", true);
                }

                BinaryFormatter bf = new BinaryFormatter();
                FileStream fs = new FileStream(EncodedModelFilePath, FileMode.Create, FileAccess.Write);
                bf.Serialize(fs, tosave);
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
            EncodedModelFilePath = modelFilePath;

            ModelAttentionData tosave = new ModelAttentionData();
            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream(EncodedModelFilePath, FileMode.Open, FileAccess.Read);
            tosave = bf.Deserialize(fs) as ModelAttentionData;
            fs.Close();
            fs.Dispose();


            this.bd = tosave.bd;
            this.clipvalue = tosave.clipval;
            this.decoder = tosave.decoder;
            this.Depth = tosave.Depth;
            this.encoder = tosave.encoder;
            this.HiddenSize = tosave.hidden_sizes;
            StartLearningRate = tosave.learning_rate;
            this.WordVectorSize = tosave.letter_size;
            this.max_word = 100;
            this.regc = tosave.regc;
            this.reversEncoder = tosave.ReversEncoder;
            this.UseDropout = tosave.UseDropout;
            this.Whd = tosave.Whd;
            this.s_Embedding = tosave.s_Wil;
            this.s_wordToIndex = tosave.s_wordToIndex;
            this.s_indexToWord = tosave.s_indexToWord;

            this.t_Embedding = tosave.t_Wil;
            this.t_wordToIndex = tosave.t_wordToIndex;
            this.t_indexToWord = tosave.t_indexToWord;

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

         
        public IWeightMatrix s_Wil;
        public IWeightMatrix t_Wil;
        public Encoder encoder;
        public Encoder ReversEncoder;
        public AttentionDecoder decoder; 
        public bool UseDropout { get; set; }


        //Output Layer Weights
        public IWeightMatrix Whd { get; set; }
        public IWeightMatrix bd { get; set; }

        public int Depth { get; set; }

        public ConcurrentDictionary<string, int> s_wordToIndex;
        public ConcurrentDictionary<int, string> s_indexToWord;

        public ConcurrentDictionary<string, int> t_wordToIndex;
        public ConcurrentDictionary<int, string> t_indexToWord;
    }
}
