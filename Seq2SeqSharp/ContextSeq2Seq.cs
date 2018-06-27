

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp
{
    public class ContextSeq2Seq
    {

        public event EventHandler IterationDone;

        public int max_word = 100; // max length of generated sentences 
        public Dictionary<string, int> wordToIndex = new Dictionary<string, int>(); 
        public Dictionary<int, string> indexToWord = new Dictionary<int, string>();
        public List<string> vocab = new List<string>();
        public List<List<string>> InputSequences;
        public List<List<string>> OutputSequences;
        public int hidden_size;
        public int word_size;

        // optimization  hyperparameters
        public float regc = 0.000001f; // L2 regularization strength
        public float learning_rate = 0.001f; // learning rate
        public float clipval = 50.0f; // clip gradients at this value


        public Optimizer solver;
        public WeightMatrix Embedding;
        public Encoder encoder;
        public Encoder ReversEncoder;
        public ContextDecoder decoder; 

        public bool UseDropout { get; set; }


        //Output Layer Weights
        public WeightMatrix Whd { get; set; }
        public WeightMatrix bd { get; set; }

        public int Depth { get; set; }

        public ContextSeq2Seq(int inputSize, int hiddenSize, int depth, List<List<string>> input, List<List<string>> output, bool useDropout)
        {
            this.InputSequences = input;
            this.OutputSequences = output;
            this.Depth=depth;
            // list of sizes of hidden layers
            word_size = inputSize; // size of word embeddings.

            this.hidden_size = hiddenSize;
            solver = new Optimizer();

            OneHotEncoding(input, output);

            this.Whd = new WeightMatrix(hidden_size , vocab.Count + 2,  true);
            this.bd = new WeightMatrix(1, vocab.Count + 2, 0);
             
            Embedding = new WeightMatrix(vocab.Count + 2, word_size,   true);
          
            encoder = new Encoder(hidden_size, word_size, depth);
            ReversEncoder = new Encoder(hidden_size, word_size, depth);

            decoder = new ContextDecoder(hidden_size, word_size, depth);
             


        }

        private void OneHotEncoding(List<List<string>> _input, List<List<string>> _output)
        {


            // count up all words
            Dictionary<string, int> d = new Dictionary<string, int>();
            wordToIndex = new Dictionary<string, int>();
            indexToWord = new Dictionary<int, string>();
            vocab = new List<string>(); 
            for (int j = 0, n2 = _input.Count; j < n2; j++)
            {
                var item = _input[j];
                for (int i = 0, n = item.Count; i < n; i++)
                {
                    var txti = item[i];
                    if (d.Keys.Contains(txti)) { d[txti] += 1; }
                    else { d.Add(txti, 1); }
                }

                var item2 = _output[j];
                for (int i = 0, n = item2.Count; i < n; i++)
                {
                    var txti = item2[i];
                    if (d.Keys.Contains(txti)) { d[txti] += 1; }
                    else { d.Add(txti, 1); }
                }

            }

            // NOTE: start at one because we will have START and END tokens!
            // that is, START token will be index 0 in model word vectors
            // and END token will be index 0 in the next word softmax
            var q = 2;
            foreach (var ch in d)
            {
                if (ch.Value >= 1)
                {
                    // add word to vocab
                    wordToIndex[ch.Key] = q;
                    indexToWord[q] = ch.Key;
                    vocab.Add(ch.Key);
                    q++;
                }

            }

             
        }

        public void Train(int trainingEpoch)
        {
            for (int ep = 0; ep < trainingEpoch; ep++)
            {
                Random r = new Random();
                for (int itr = 0; itr < InputSequences.Count; itr++)
                {
                    // sample sentence from data
                    List<string> OutputSentence;
                    IComputeGraph g;
                    float cost;
                    List<WeightMatrix> encoded = new List<WeightMatrix>();
                    Encode(r, out OutputSentence, out g, out cost,   encoded);
                    cost = DecodeOutput(OutputSentence, g, cost, encoded);

                    g.backward();
                    UpdateParameters();
                    Reset();
                    if (IterationDone != null)
                    {
                        IterationDone(this, new CostEventArg() { Cost = cost / OutputSentence.Count});
                    }
                     
                }
            }

        }

        private void Encode(Random r, out List<string> OutputSentence, out IComputeGraph g, out float cost,   List<WeightMatrix> encoded)
        {
            var sentIndex = r.Next(0, InputSequences.Count);
            var inputSentence = InputSequences[sentIndex];
            var reversSentence = InputSequences[sentIndex].ToList();
            reversSentence.Reverse();
            OutputSentence = OutputSequences[sentIndex];
            g = new ComputeGraph();

            cost = 0.0f;          
            for (int i = 0; i < inputSentence.Count; i++)
            {
                int ix_source = wordToIndex[inputSentence[i]];
                int ix_source2 = wordToIndex[reversSentence[i]];
                var x = g.PeekRow(Embedding, ix_source);
                var eOutput = encoder.Encode(x, g);
                var x2 = g.PeekRow(Embedding, ix_source2);
                var eOutput2 = ReversEncoder.Encode(x2, g);
                encoded.Add( g.concatColumns(eOutput, eOutput2));

            }


            //if (UseDropout)
            //{
            //    encoded = g.Dropout(encoded, 0.2);
            //}
        }

        private float DecodeOutput(List<string> OutputSentence, IComputeGraph g, float cost, List<WeightMatrix> encoded)
        {
            int ix_input = 1;
            for (int i = 0; i < OutputSentence.Count + 1; i++)
            {
                int ix_target = 0;
                if (i == OutputSentence.Count)
                {
                    ix_target = 0;
                }
                else
                {
                    ix_target = wordToIndex[OutputSentence[i]];
                }


                var x = g.PeekRow(Embedding, ix_input);
                var eOutput = decoder.Decode(x, encoded.LastOrDefault(), g);
                if (UseDropout)
                {
                    eOutput = g.Dropout(eOutput, 0.2f);

                }
                var o = g.muladd(eOutput, this.Whd, this.bd);
                if (UseDropout)
                {
                    o = g.Dropout(o, 0.2f);

                }

                var probs = g.SoftmaxWithCrossEntropy(o);

                cost += (float)(-Math.Log(probs.Weight[ix_target]));
                o.Gradient = probs.Weight;
                o.Gradient[ix_target] -= 1;
                ix_input = ix_target;
            }
            return cost;
        }

        private void UpdateParameters()
        {
            var model = encoder.getParams();
            model.AddRange(decoder.getParams());
            model.AddRange(ReversEncoder.getParams());
            model.Add(Embedding);
            model.Add(Whd);
            model.Add(bd);
            solver.setp(model, learning_rate, regc, clipval);
        }
        
        private void Reset()
        {
            encoder.Reset();
            ReversEncoder.Reset();
            decoder.Reset();
        }


         
        public List<string> Predict(List<string> inputSeq)
        {
            ReversEncoder.Reset();
            encoder.Reset();
            decoder.Reset();
          
            List<string> result = new List<string>();

            var G2 = new ComputeGraph(false);



            List<string> revseq = inputSeq.ToList();
            revseq.Reverse();
            List<WeightMatrix>  encoded = new List<WeightMatrix>(); 
            for (int i = 0; i < inputSeq.Count; i++)
            { 
                int  ix = wordToIndex[inputSeq[i]];
                int  ix2 = wordToIndex[revseq[i]]; 
                var x2 = G2.PeekRow(Embedding, ix);
                var o = encoder.Encode(x2, G2);
                var x3 = G2.PeekRow(Embedding, ix2);
                var eOutput2 = ReversEncoder.Encode(x3, G2); 
              
                    var d = G2.concatColumns(o, eOutput2);
                   
                    encoded.Add(d);
                 
            }
             
            var ix_input = 1;
            while(true)
            {
                var x = G2.PeekRow(Embedding, ix_input);
                var eOutput = decoder.Decode(x,encoded.LastOrDefault(), G2);
                if (UseDropout)
                {
                    for (int i = 0; i < eOutput.Weight.Length; i++)
                    {
                        eOutput.Weight[i] *= 0.2f;
                    }
                } 
                var o = G2.muladd(eOutput, this.Whd, this.bd);
                if (UseDropout)
                {
                    for (int i = 0; i < o.Weight.Length; i++)
                    {
                        o.Weight[i] *= 0.2f;
                    }
                }
                var probs = G2.SoftmaxWithCrossEntropy(o);
                var maxv = probs.Weight[0];
                var maxi = 0;
                for (int i = 1; i < probs.Weight.Length; i++)
                {
                    if (probs.Weight[i] > maxv)
                    {
                        maxv = probs.Weight[i];
                        maxi = i;
                    }
                }
                var pred = maxi;

                if (pred == 0) break; // END token predicted, break out
                
                if (result.Count > max_word) { break; } // something is wrong 
                var letter2 = indexToWord[pred];
                result.Add(letter2);
                ix_input = pred;
            }

            return result;
        }

        public void Save()
        {

            ModelContextData tosave = new ModelContextData(); 
            tosave.bd = this.bd;
            tosave.clipval = this.clipval;
            tosave.decoder = this.decoder;
            tosave.Depth = this.Depth;
            tosave.encoder = this.encoder;
            tosave.hidden_sizes = this.hidden_size;
            tosave.learning_rate = this.learning_rate;
            tosave.letter_size = this.word_size;
            tosave.max_chars_gen = this.max_word;
            tosave.regc = this.regc;
            tosave.ReversEncoder = this.ReversEncoder;
            tosave.UseDropout = this.UseDropout;
            tosave.Whd = this.Whd;
            tosave.Wil = this.Embedding;

            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream("Model.bin", FileMode.OpenOrCreate, FileAccess.Write);
            bf.Serialize(fs, tosave);
            fs.Close();
            fs.Dispose();


        }
        public void Load()
        {
            ModelContextData tosave = new ModelContextData();
            BinaryFormatter bf = new BinaryFormatter();
            FileStream fs = new FileStream("Model.bin", FileMode.Open, FileAccess.Read);
            tosave = bf.Deserialize(fs) as ModelContextData;
            fs.Close();
            fs.Dispose();


            this.bd = tosave.bd;
            this.clipval = tosave.clipval;
            this.decoder = tosave.decoder;
            this.Depth = tosave.Depth;
            this.encoder = tosave.encoder;
            this.hidden_size = tosave.hidden_sizes;
            this.learning_rate = tosave.learning_rate;
            this.word_size = tosave.letter_size;
            this.max_word = 100;
            this.regc = tosave.regc;
            this.ReversEncoder = tosave.ReversEncoder;
            this.UseDropout = tosave.UseDropout;
            this.Whd = tosave.Whd;
            this.Embedding = tosave.Wil;

        }
    }

    [Serializable]
    public class ModelContextData
    {

        public int max_chars_gen = 100; // max length of generated sentences  
        public int hidden_sizes;
        public int letter_size;

        // optimization  
        public float regc = 0.000001f; // L2 regularization strength
        public float learning_rate = 0.01f; // learning rate
        public float clipval = 5.0f; // clip gradients at this value

         
        public WeightMatrix Wil;
        public Encoder encoder;
        public Encoder ReversEncoder;
        public ContextDecoder decoder; 
        public bool UseDropout { get; set; }


        //Output Layer Weights
        public WeightMatrix Whd { get; set; }
        public WeightMatrix bd { get; set; }

        public int Depth { get; set; }
    }
}
