

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
    public class RoundArray<T>
    {
        T[] m_array;
        int currentIdx = 0;
        public RoundArray(T[] a)
        {
            m_array = a;
        }

        public T GetNextItem()
        {
            T item = m_array[currentIdx];
            currentIdx = (currentIdx + 1) % m_array.Length;

            return item;
        }
    }

    public class AttentionSeq2Seq : BaseSeq2SeqFramework
    {
        Seq2SeqModelMetaData m_modelMetaData;

        // Trainable parameters including networks and tensors
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<AttentionDecoder> m_decoder; //The LSTM decoders over devices        
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after LSTM layers in decoder

        // optimization  hyperparameters
        private float m_dropoutRatio = 0.0f;
        private string m_modelFilePath;
        private int m_defaultDeviceId = 0;

        public AttentionSeq2Seq(string modelFilePath, ProcessorTypeEnums processorType, int[] deviceIds, Corpus trainCorpus = null, float dropoutRatio = 0.0f, float gradClip = 3.0f)
            :base(trainCorpus, deviceIds, gradClip, processorType, modelFilePath)
        {
            m_dropoutRatio = dropoutRatio;
            m_modelMetaData = LoadModel(CreateTrainableParameters) as Seq2SeqModelMetaData;           
        }

        public AttentionSeq2Seq(int embeddingDim, int hiddenDim, int encoderLayerDepth, int decoderLayerDepth, Corpus trainCorpus, Vocab vocab, string srcEmbeddingFilePath, string tgtEmbeddingFilePath, 
            string modelFilePath, float dropoutRatio, int multiHeadNum, float gradClip, ProcessorTypeEnums processorType, EncoderTypeEnums encoderType, int[] deviceIds)
            :base(trainCorpus, deviceIds, gradClip, processorType, modelFilePath)
        {
            m_modelMetaData = new Seq2SeqModelMetaData(hiddenDim, embeddingDim, encoderLayerDepth, decoderLayerDepth, multiHeadNum, encoderType, vocab);
            m_dropoutRatio = dropoutRatio;

            //Initializng weights in encoders and decoders
            CreateTrainableParameters(m_modelMetaData);

            // Load external embedding from files
            for (int i = 0; i < DeviceIds.Length; i++)
            {
                //If pre-trained embedding weights are speicifed, loading them from files
                if (!String.IsNullOrEmpty(srcEmbeddingFilePath))
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{srcEmbeddingFilePath}' for source side.");
                    LoadWordEmbedding(srcEmbeddingFilePath, m_srcEmbedding.GetNetworkOnDevice(i), m_modelMetaData.Vocab.SrcWordToIndex);
                }

                if (!String.IsNullOrEmpty(tgtEmbeddingFilePath))
                {
                    Logger.WriteLine($"Loading ExtEmbedding model from '{tgtEmbeddingFilePath}' for target side.");
                    LoadWordEmbedding(tgtEmbeddingFilePath, m_tgtEmbedding.GetNetworkOnDevice(i), m_modelMetaData.Vocab.TgtWordToIndex);
                }
            }
        }

        private bool CreateTrainableParameters(IModelMetaData mmd)
        {
            Logger.WriteLine($"Creating encoders and decoders...");
            Seq2SeqModelMetaData modelMetaData = mmd as Seq2SeqModelMetaData;
            RoundArray<int> raDeviceIds = new RoundArray<int>(DeviceIds);

            if (modelMetaData.EncoderType == EncoderTypeEnums.BiLSTM)
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new BiEncoder("BiLSTMEncoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, raDeviceIds.GetNextItem()), DeviceIds);
                m_decoder = new MultiProcessorNetworkWrapper<AttentionDecoder>(
                    new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.HiddenDim * 2, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem()), DeviceIds);
            }
            else
            {
                m_encoder = new MultiProcessorNetworkWrapper<IEncoder>(
                    new TransformerEncoder("TransformerEncoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.EncoderLayerDepth, m_dropoutRatio, raDeviceIds.GetNextItem()), DeviceIds);
                m_decoder = new MultiProcessorNetworkWrapper<AttentionDecoder>(
                    new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.EmbeddingDim, modelMetaData.HiddenDim, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem()), DeviceIds);
            }
            m_srcEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.SourceWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: true, name: "SrcEmbeddings", isTrainable: true), DeviceIds);
            m_tgtEmbedding = new MultiProcessorNetworkWrapper<IWeightTensor>(new WeightTensor(new long[2] { modelMetaData.Vocab.TargetWordSize, modelMetaData.EmbeddingDim }, raDeviceIds.GetNextItem(), normal: true, name: "TgtEmbeddings", isTrainable: true), DeviceIds);
            m_decoderFFLayer = new MultiProcessorNetworkWrapper<FeedForwardLayer>(new FeedForwardLayer("FeedForward", modelMetaData.HiddenDim, modelMetaData.Vocab.TargetWordSize, dropoutRatio: 0.0f, deviceId: raDeviceIds.GetNextItem()), DeviceIds);

            return true;
        }


        private void LoadWordEmbedding(string extEmbeddingFilePath, IWeightTensor embeddingMatrix, IEnumerable<KeyValuePair<string, int>> wordToIndex)
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

        public void Train(int maxTrainingEpoch, ILearningRate learningRate)
        {
            Logger.WriteLine("Start to train...");
            for (int i = 0; i < maxTrainingEpoch; i++)
            {
                // Train one epoch over given devices. Forward part is implemented in RunForwardOnSingleDevice function in below, 
                // backward, weights updates and other parts are implemented in the framework. You can see them in BaseSeq2SeqFramework.cs
                TrainOneEpoch(i, learningRate, new Optimizer(), m_modelMetaData, RunForwardOnSingleDevice);
            }
        }

        /// <summary>
        /// Get networks on specific devices
        /// </summary>
        /// <param name="deviceIdIdx"></param>
        /// <returns></returns>
        private (IEncoder, AttentionDecoder, IWeightTensor, IWeightTensor, FeedForwardLayer) GetNetworksOnDeviceAt(int deviceIdIdx)
        {
            return (m_encoder.GetNetworkOnDevice(deviceIdIdx), m_decoder.GetNetworkOnDevice(deviceIdIdx), m_srcEmbedding.GetNetworkOnDevice(deviceIdIdx), 
                m_tgtEmbedding.GetNetworkOnDevice(deviceIdIdx), m_decoderFFLayer.GetNetworkOnDevice(deviceIdIdx));
        }

        /// <summary>
        /// Run forward part on given single device
        /// </summary>
        /// <param name="computeGraph">The computing graph for current device. It gets created and passed by the framework</param>
        /// <param name="srcSnts">A batch of input tokenized sentences in source side</param>
        /// <param name="tgtSnts">A batch of output tokenized sentences in target side</param>
        /// <param name="deviceIdIdx">The index of current device</param>
        /// <returns>The cost of forward part</returns>
        private float RunForwardOnSingleDevice(IComputeGraph computeGraph, List<List<string>> srcSnts, List<List<string>> tgtSnts, int deviceIdIdx)
        {   
            (IEncoder encoder, AttentionDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(deviceIdIdx);

            // Reset networks
            encoder.Reset(computeGraph.GetWeightFactory(), srcSnts.Count);
            decoder.Reset(computeGraph.GetWeightFactory(), tgtSnts.Count);

            // Encoding input source sentences
            IWeightTensor encodedWeightMatrix = Encode(computeGraph.CreateSubGraph("Encoder"), srcSnts, encoder, srcEmbedding);
            // Generate output decoder sentences
            return Decode(tgtSnts, computeGraph.CreateSubGraph("Decoder"), encodedWeightMatrix, decoder, decoderFFLayer, tgtEmbedding);
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
        private IWeightTensor Encode(IComputeGraph g, List<List<string>> inputSentences, IEncoder encoder, IWeightTensor Embedding)
        {
            Corpus.PadSentences(inputSentences);
            int seqLen = inputSentences[0].Count;
            int batchSize = inputSentences.Count;

            List<IWeightTensor> forwardInput = new List<IWeightTensor>();
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < inputSentences.Count; j++)
                {
                    int ix_source = m_modelMetaData.Vocab.GetSourceWordIndex(inputSentences[j][i], logUnk: true);
                    forwardInput.Add(g.PeekRow(Embedding, ix_source));
                }
            }

            return encoder.Encode(g.ConcatRows(forwardInput), batchSize, g);
        }

        /// <summary>
        /// Decode output sentences in training
        /// </summary>
        /// <param name="outputSentences"></param>
        /// <param name="g"></param>
        /// <param name="encodedOutputs"></param>
        /// <param name="decoder"></param>
        /// <param name="decoderFFLayer"></param>
        /// <param name="Embedding"></param>
        /// <returns></returns>
        private float Decode(List<List<string>> outputSentences, IComputeGraph g, IWeightTensor encodedOutputs, AttentionDecoder decoder, FeedForwardLayer decoderFFLayer, IWeightTensor Embedding)
        {
            float cost = 0.0f;
            int batchSize = outputSentences.Count;
            var originalOutputLengths = Corpus.PadSentences(outputSentences);
            int seqLen = outputSentences[0].Count;
            int[] ix_inputs = new int[batchSize];

            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = (int)SENTTAGS.START;
            }

            var attPreProcessResult = decoder.PreProcess(encodedOutputs, batchSize, g);
            for (int i = 0; i < seqLen; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    inputs.Add(g.PeekRow(Embedding, ix_inputs[j]));
                }           
                var inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                var eOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);
                eOutput = g.Dropout(eOutput, batchSize, m_dropoutRatio, true);                
                eOutput = decoderFFLayer.Process(eOutput, batchSize, g);

                //Softmax for output
                using (var probs = g.Softmax(eOutput, runGradients: false, inPlace: true))
                {
                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        using (var probs_k = g.PeekRow(probs, k, runGradients: false))
                        {
                            var ix_targets_k = m_modelMetaData.Vocab.GetTargetWordIndex(outputSentences[k][i]);
                            var score_k = probs_k.GetWeightAt(ix_targets_k);
                            if (i < originalOutputLengths[k])
                            {
                                cost += (float)-Math.Log(score_k);
                            }

                            probs_k.SetWeightAt(score_k - 1, ix_targets_k);
                            ix_inputs[k] = ix_targets_k;
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

        /// <summary>
        /// Given input sentence and generate output sentence by seq2seq model with beam search
        /// </summary>
        /// <param name="input"></param>
        /// <param name="beamSearchSize"></param>
        /// <param name="maxOutputLength"></param>
        /// <returns></returns>
        public List<List<string>> Predict(List<string> input, int beamSearchSize = 1, int maxOutputLength = 100)
        {
            (IEncoder encoder, AttentionDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(-1);
            var inputSeqs = Corpus.ConstructInputSentence(input);
            int batchSize = inputSeqs.Count;

            var g = CreateComputGraph(m_defaultDeviceId, needBack: false);
            encoder.Reset(g.GetWeightFactory(), batchSize);
            decoder.Reset(g.GetWeightFactory(), batchSize);

            // Construct beam search status list
            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();
            BeamSearchStatus bss = new BeamSearchStatus();
            bss.OutputIds.Add((int)SENTTAGS.START);
            bss.CTs = decoder.GetCTs();
            bss.HTs = decoder.GetHTs();
            bssList.Add(bss);

            IWeightTensor encodedWeightMatrix = Encode(g.CreateSubGraph("Encoder"), inputSeqs, encoder, srcEmbedding);

            g = g.CreateSubGraph("Decoder");
            var attPreProcessResult = decoder.PreProcess(encodedWeightMatrix, batchSize, g);

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
                        using (var probs = g.Softmax(o))
                        {
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
                }

                bssList = BeamSearch.GetTopNBSS(newBSSList, beamSearchSize);
                newBSSList.Clear();

                outputLength++;
            }
            
            // Convert output target word ids to real string
            List<List<string>> results = new List<List<string>>();
            for (int i = 0; i < bssList.Count; i++)
            {
                results.Add(m_modelMetaData.Vocab.ConvertTargetIdsToString(bssList[i].OutputIds));                
            }

            return results;
        }

        public void VisualizeNeuralNetwork(string visNNFilePath)
        {
            (IEncoder encoder, AttentionDecoder decoder, IWeightTensor srcEmbedding, IWeightTensor tgtEmbedding, FeedForwardLayer decoderFFLayer) = GetNetworksOnDeviceAt(-1);
            // Build input sentence
            var inputSeqs = Corpus.ConstructInputSentence(null);
            int batchSize = inputSeqs.Count;
            var g = CreateComputGraph(m_defaultDeviceId, needBack: false, visNetwork: true);

            encoder.Reset(g.GetWeightFactory(), batchSize);
            decoder.Reset(g.GetWeightFactory(), batchSize);

            // Run encoder
            IWeightTensor encodedWeightMatrix = Encode(g.CreateSubGraph("Encoder"), inputSeqs, encoder, srcEmbedding);

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
    }
}
