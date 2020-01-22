# Seq2SeqSharp  
Seq2SeqSharp is a tensor based fast & flexible encoder-decoder deep neural network framework written by .NET (C#). It can be used for sequence-to-sequence task, sequence-labeling task and sequence-classification task and other NLP tasks. Seq2SeqSharp supports both CPUs and GPUs.  

# Features  
Pure C# framework   
Bi-directional LSTM encoder  
Attention based LSTM decoder with coverage model  
Transformer encoder  
Built-in several networks for sequence-to-sequence task and sequence-labeling task  
Graph based neural network  
Automatic differentiation  
Tensor based operations  
Running on both CPUs and GPUs (CUDA)  
Support multi-GPUs  
Mini-batch  
Dropout  
RMSProp and Adam optmization  
Embedding & Pre-trained model 
Metrics, such as BLEU score, Length ratio, F1 score and so on  
Auto data shuffling  
Auto vocabulary building  
Beam search decoder  
Visualize neural network  

# Architecture  
Here is the architecture of Seq2SeqSharp  
![](https://raw.githubusercontent.com/zhongkaifu/Seq2SeqSharp/master/Overview.jpg)

Seq2SeqSharp provides the unified tensor operations, which means all tensor operations running on CPUs and GPUs are completely same and they can get switched on different device types without any modification.  
Seq2SeqSharp is also a framework that neural networks can run on multi-GPUs in parallel. It can automatically distribute/sync weights/gradients over devices, manage resources and models and so on, so developers are able to totally focus on how to design and implment networks for their tasks.  

# Usage  
Seq2SeqSharp provides two console tools that you can run for sequence-to-sequence task (**Seq2SeqConsole.exe**) and sequence-labeling task (**SeqLabelConsole.exe**).  

You can use Seq2SeqConsole tool to train, test and visualize models.  
Here is the command line to train a model:  
**Seq2SeqConsole.exe -TaskName Train [parameters...]**  
Parameters:  
**-WordVectorSize**: The vector size of encoded source word.  
**-HiddenSize**: The hidden layer size of encoder and decoder.    
**-LearningRate**: Learning rate. Default is 0.001  
**-EncoderLayerDepth**: The network depth in encoder. The default depth is 1.  
**-DecoderLayerDepth**: The network depth in decoder. The default depth is 1.  
**-EncoderType**: The type of encoder. It supports BiLSTM and Transformer.  
**-MultiHeadNum**: The number of multi-heads in Transformer encoder.  
**-ModelFilePath**: The trained model file path.  
**-SrcVocab**: The vocabulary file path for source side.  
**-TgtVocab**: The vocabulary file path for target side.  
**-SrcEmbedding**: The external embedding model file path for source side. It is built by Txt2Vec project.  
**-TgtEmbedding**: The external embedding model file path for target side. It is built by Txt2Vec project.  
**-SrcLang**: Source language name.  
**-TgtLang**: Target language name.  
**-TrainCorpusPath**: training corpus folder path  
**-ValidCorpusPath**: valid corpus folder path  
**-ShuffleBlockSize**: The block size for corpus shuffle. The default value is -1 which means we shuffle entire corpus.  
**-GradClip**: The clip gradients.  
**-BatchSize**: Mini-batch size. Default is 1.  
**-Dropout**: Dropout ratio. Defaul is 0.1  
**-ProcessorType**: Processor type: CPU or GPU  
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-MaxEpochNum**: Maxmium epoch number during training. Default is 100  
**-MaxSentLength**: Maxmium sentence length  
**-WarmUpSteps**: The number of steps for warming up. Default is 8,000  
Note that:  
  1) if "-SrcVocab" and "-TgtVocab" are empty, vocabulary will be built from training corpus.  
  2) Txt2Vec for external embedding model building can get downloaded from https://github.com/zhongkaifu/Txt2Vec  

Example: Seq2SeqConsole.exe -TaskName Train -WordVectorSize 512 -HiddenSize 512 -LearningRate 0.002 -ModelFilePath seq2seq.model -TrainCorpusPath .\corpus -ValidCorpusPath .\corpus_valid -SrcLang ENU -TgtLang CHS -BatchSize 256 -ProcessorType GPU -EncoderType Transformer -EncoderLayerDepth 6 -DecoderLayerDepth 2 -MultiHeadNum 8 -DeviceIds 0,1,2,3,4,5,6,7  

During training, the iteration information will be printed out and logged as follows:  
info,9/26/2019 3:38:24 PM Update = '15600' Epoch = '0' LR = '0.002000', Current Cost = '2.817434', Avg Cost = '3.551963', SentInTotal = '31948800', SentPerMin = '52153.52', WordPerSec = '39515.27'  
info,9/26/2019 3:42:28 PM Update = '15700' Epoch = '0' LR = '0.002000', Current Cost = '2.800056', Avg Cost = '3.546863', SentInTotal = '32153600', SentPerMin = '52141.86', WordPerSec = '39523.83'  

Here is the command line to valid models  
**Seq2SeqConsole.exe -TaskName Valid [parameters...]**  
Parameters:  
**-ModelFilePath**: The trained model file path.  
**-SrcLang**: Source language name.  
**-TgtLang**: Target language name.  
**-ValidCorpusPath**: valid corpus folder path  

Example: Seq2SeqConsole.exe -TaskName Valid -ModelFilePath seq2seq.model -SrcLang ENU -TgtLang CHS -ValidCorpusPath .\corpus_valid  

Here is the command line to test models  
**Seq2SeqConsole.exe -TaskName Test [parameters...]**  
Parameters:  
**-InputTestFile**: The input file for test.  
**-OutputTestFile**: The test result file.  
**-ModelFilePath**: The trained model file path. 
**-ProcessorType**: Architecture type: CPU or GPU 
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-BeamSearch**: Beam search size. Default is 1  

Example: Seq2SeqConsole.exe -TaskName Test -ModelFilePath seq2seq.model -InputTestFile test.txt -OutputTestFile result.txt -ProcessorType CPU -BeamSearch 5  

Here is the command line to visualize network  
**Seq2SeqConsole.exe -TaskName VisualizeNetwork [parameters...]**  
Parameters:  
**-VisNNFile**: The output PNG file to visualize network  
**-EncoderType**: The type of encoder. BiLSTM and Transformer are built-in and you can implement your own network and visualize it  
**-EncoderLayerDepth**: The network depth in encoder. The default depth is 1.  
**-DecoderLayerDepth**: The network depth in decoder. The default depth is 1.  

Example: Seq2SeqConsole.exe -TaskName VisualizeNetwork -VisNNFile abc.png -EncoderType Transformer -EncoderLayerDepth 2 -DecoderLayerDepth 2  

Then it will visualize the network looks like below:  
![](https://raw.githubusercontent.com/zhongkaifu/Seq2SeqSharp/master/NetworkViz.png)

You can also keep all parameters into a json file and run Seq2SeqConsole.exe -ConfigFilePath <config_file_path> Here is an example for training.  
```json
{
"TaskName":"Train",
"WordVectorSize":1024,
"HiddenSize":1024,
"StartLearningRate":0.001,
"WeightsUpdateCount":0,
"EncoderLayerDepth":6,
"DecoderLayerDepth":6,
"ModelFilePath":"seq2seq.model",
"SrcVocab":"corpus\\vocab.enu",
"TgtVocab":"corpus\\vocab.chs",
"SrcEmbeddingModelFilePath":null,
"TgtEmbeddingModelFilePath":null,
"SrcLang":"ENU",
"TgtLang":"CHS",
"TrainCorpusPath":"corpus",
"ValidCorpusPath":"corpus_valid",
"InputTestFile":null,
"OutputTestFile":null,
"ShuffleBlockSize":-1,
"GradClip":3.0,
"BatchSize":128,
"DropoutRatio":0.1,
"ProcessorType":"GPU",
"EncoderType":"Transformer",
"MultiHeadNum":16,
"DeviceIds":"0,1,2,3",
"BeamSearch":1,
"MaxEpochNum":100,
"MaxSentLength":64,
"WarmUpSteps":8000,
"VisualizeNNFilePath":null,
"Beta1":0.9,
"Beta2":0.98,
"EnableCoverageModel":true
}
```

The usage of **SeqLabelConsole.exe** is similar as **Seq2SeqConsole.exe** in above, you can just type it in the console and it will show you usage.  

# Data Format  
The corpus contains each sentence per line. The file name pattern is "mainfilename.{source language name}.snt" and "mainfilename.{target language name}.snt".    
For example: Let's use three letters name CHS for Chinese and ENU for English in Chinese-English parallel corpus, so we could have these corpus files: train01.enu.snt, train01.chs.snt, train02.enu.snt and train02.chs.snt.  
In train01.enu.snt, assume we have below two sentences:  
the children huddled together for warmth .  
the car business is constantly changing .  
So, train01.chs.snt has the corresponding translated sentences:  
孩子 们 挤 成 一 团 以 取暖 .  
汽车 业 也 在 不断 地 变化 .  

For sequence-labeling task, the corpus format is the same as above. The target corpus contains labels for the corresponding sentences in the source corpus.  
For example:  
In train01.word.snt, assume we have below two sentences:  
Microsoft is located in Redmond .  
Zhongkai Fu is the author of Seq2SeqSharp .  
In train01.label.snt, we will have the following label sequences:  
S_ORG S_NOR S_NOR S_NOR S_LOC S_NOR  
B_PER E_PER S_NOR S_NOR S_NOR S_NOR S_NOR S_NOR  

# Build Your Layers  
Benefit from automatic differentiation, tensor based compute graph and other features, you can easily build your customized layers by a few code. The only thing you need to implment is forward part, and the framework will automatically build the corresponding backward part for you, and make the network could run on multi-GPUs or CPUs.  
Here is an example about **attentioned based LSTM cells**.  
```c#
        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public IWeightTensor Step(IWeightTensor context, IWeightTensor input, IComputeGraph g)
        {
            var computeGraph = g.CreateSubGraph(m_name);

            var cell_prev = Cell;
            var hidden_prev = Hidden;

            var hxhc = computeGraph.ConcatColumns(input, hidden_prev, context);
            var hhSum = computeGraph.Affine(hxhc, m_Wxhc, m_b);
            var hhSum2 = layerNorm1.Process(hhSum, computeGraph);

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum2, m_hdim * 3, m_hdim);
            var gates = computeGraph.Sigmoid(gates_raw);
            var cell_write = computeGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, m_hdim, m_hdim, m_hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            Cell = computeGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(Cell, computeGraph);

            Hidden = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct2));

            return Hidden;
        }
```
Another example about **scaled multi-heads attention** component which is the core part in **Transformer** model.  
```c#
        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="g">The instance of computing graph</param>
        /// <returns></returns>
        public IWeightTensor Perform(IWeightTensor input, IComputeGraph graph)
        {
            IComputeGraph g = graph.CreateSubGraph(m_name);

            var seqLen = input.Rows / m_batchSize;

            //Input projections
            var allQ = g.View(Q.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);
            var allK = g.View(K.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);
            var allV = g.View(V.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);

            //Multi-head attentions
            var Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);
            var Ks = g.View(g.Permute(allK, 2, 0, 3, 1), m_multiHeadNum * m_batchSize, m_d, seqLen);
            var Vs = g.View(g.Permute(allV, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);

            // Scaled softmax
            float scale = 1.0f / (float)Math.Sqrt(m_d);
            var attn = g.MulBatch(Qs, Ks, m_multiHeadNum * m_batchSize, scale);
            var attn2 = g.View(attn, m_multiHeadNum * m_batchSize * seqLen, seqLen);

            var softmax = g.Softmax(attn2);
            var softmax2 = g.View(softmax, m_multiHeadNum * m_batchSize, seqLen, seqLen);
            var o = g.View(g.MulBatch(softmax2, Vs, m_multiHeadNum * m_batchSize), m_multiHeadNum, m_batchSize, seqLen, m_d);
            var W = g.View(g.Permute(o, 1, 2, 0, 3), m_batchSize * seqLen, m_multiHeadNum * m_d);

            // Output projection
            var finalAttResults = g.Affine(W, W0, b0);

            //Skip connection and layer normaliztion
            var addedAttResult = g.Add(finalAttResults, input);
            var normAddedAttResult = layerNorm1.Process(addedAttResult, g);

            //Feed forward
            var ffnResult = feedForwardLayer1.Process(normAddedAttResult, g);
            var reluFFNResult = g.Relu(ffnResult);
            var ffn2Result = feedForwardLayer2.Process(reluFFNResult, g);

            //Skip connection and layer normaliztion
            var addFFNResult = g.Add(ffn2Result, normAddedAttResult);
            var normAddFFNResult = layerNorm2.Process(addFFNResult, g);

            return normAddFFNResult;
        }
```
# Build Your Operations  
Seq2SeqSharp includes many built-in operations for neural networks. You can visit IComputeGraph.cs to get interfaces and ComputeGraphTensor.cs to get implementation.  
You can also implement your customized operations. Here is an example for "w1 * w2 + w3 * w4" in a single operation. The forward part includes 1) create result tensor and 2) call inner-operation "Ops.MulMulAdd".  
And the backward part is in "backward" action that the gradients of each input tensor(w?) will be added by the product between weights of input tensor(w?) and gradients of the output tensor(res).  
If the operations is for forward part only, you can completely ignore "backward" action.  

```c#
        public IWeightTensor EltMulMulAdd(IWeightTensor w1, IWeightTensor w2, IWeightTensor w3, IWeightTensor w4)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var m4 = w4 as WeightTensor;

            var res = m_weightTensorFactory.CreateWeightTensor(m1.Sizes, m_deviceId, name: $"{GetHashString(w1.Name, w2.Name, w3.Name, w4.Name)}.EltMulMulAdd");
            VisualizeNodes(new IWeightTensor[] { w1, w2, w3, w4 }, res);

            Ops.MulMulAdd(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);
            if (m_needsBackprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    m3.AddMulGradient(m4.TWeight, res.TGradient);
                    m4.AddMulGradient(m3.TWeight, res.TGradient);

                    res.Dispose();
                };
                this.m_backprop.Add(backward);
            }

            return res;
        }
```

# Build Your Networks  
Besides operations and layers, you can also build your customized networks by leveraging BaseSeq2SeqFramework. The built-in AttentionSeq2Seq is a good example to show you how to do it. Basically, it includes the follows steps:  
1. Define model meta data, such as hidden layer dimension, embedding diemnsion, layer depth and so on. It should be inherited from IModelMetaData interface. You can look at Seq2SeqModelMetaData.cs as an example.  
```c#
 public class Seq2SeqModelMetaData : IModelMetaData
 {
        public int HiddenDim;
        public int EmbeddingDim;
        public int EncoderLayerDepth;
        public int DecoderLayerDepth;
        public int MultiHeadNum;
        public EncoderTypeEnums EncoderType;
        public Vocab Vocab;
 }
```
2. Create the class for your network and make sure it is inherited from BaseSeq2SeqFramework class at first, and then define layers, tensors for your network. Seq2SeqSharp has some built-in layers, so you can just use them or create your customized layers by instruction in above. In order to support multi-GPUs, these layers and tensors should be wrapped by MultiProcessorNetworkWrapper class. Here is an example:  
```c#
        private MultiProcessorNetworkWrapper<IWeightTensor> m_srcEmbedding; //The embeddings over devices for target
        private MultiProcessorNetworkWrapper<IWeightTensor> m_tgtEmbedding; //The embeddings over devices for source
        private MultiProcessorNetworkWrapper<IEncoder> m_encoder; //The encoders over devices. It can be LSTM, BiLSTM or Transformer
        private MultiProcessorNetworkWrapper<AttentionDecoder> m_decoder; //The LSTM decoders over devices        
        private MultiProcessorNetworkWrapper<FeedForwardLayer> m_decoderFFLayer; //The feed forward layers over devices after LSTM layers in decoder
```
3. Initialize those layers and tensors your defined in above. You should pass variables you defined in model meta data to the constructors of layers and tensors. Here is an example in AttentionSeq2Seq.cs  
```c#
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
```
4. Implement forward part only for your network and the BaseSeq2SeqFramework will handle all other things, such as backward propagation, parameters updates, memory management, computing graph managment, corpus shuffle & batching, saving/loading for model, logging & monitoring, checkpoints and so on. Here is an example in AttentionSeq2Seq.cs as well.  
```c#
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
```
Now you already have your customized network and you can play it. See Progream.cs in Seq2SeqConsole project about how to load corpus and vocabulary, and create network for training.  

# How To Play Your Network  
In Seq2SeqConsole project, it shows you how to initialize and play (train, valid or test) your network. Here are few steps about how to do it.  
```c#
                // Load train corpus
                Corpus trainCorpus = new Corpus(opts.TrainCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);
                // Load valid corpus
                Corpus validCorpus = new Corpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.BatchSize, opts.ShuffleBlockSize, opts.MaxSentLength);

                // Load or build vocabulary
                Vocab vocab = null;
                if (!String.IsNullOrEmpty(opts.SrcVocab) && !String.IsNullOrEmpty(opts.TgtVocab))
                {
                    // Vocabulary files are specified, so we load them
                    vocab = new Vocab(opts.SrcVocab, opts.TgtVocab);
                }
                else
                {
                    // We don't specify vocabulary, so we build it from train corpus
                    vocab = new Vocab(trainCorpus);
                }

                // Create learning rate
                ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

                // Create optimizer
                Optimizer optimizer = new Optimizer(opts.GradClip);

                // Create metrics
                List<IMetric> metrics = new List<IMetric>();
                metrics.Add(new BleuMetric());
                metrics.Add(new LengthRatioMetric());

                if (File.Exists(opts.ModelFilePath) == false)
                {
                    //New training
                    ss = new AttentionSeq2Seq(embeddingDim: opts.WordVectorSize, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth,
                        srcEmbeddingFilePath: opts.SrcEmbeddingModelFilePath, tgtEmbeddingFilePath: opts.TgtEmbeddingModelFilePath, vocab: vocab, modelFilePath: opts.ModelFilePath, 
                        dropoutRatio: opts.DropoutRatio, processorType: processorType, deviceIds: deviceIds, multiHeadNum: opts.MultiHeadNum, encoderType: encoderType);
                }
                else
                {
                    //Incremental training
                    Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
                    ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, processorType: processorType, dropoutRatio: opts.DropoutRatio, deviceIds: deviceIds);
                }

                // Add event handler for monitoring
                ss.IterationDone += ss_IterationDone;

                // Kick off training
                ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, metrics: metrics);
```

# Todo List  
If you are interested in below items, please let me know. Becuase African proverb says "If you want to go fast, go alone. If you want to go far, go together" :)  
Support Tensor Cores in CUDA  
Support Half-Float Type (FP16)  
And More...  