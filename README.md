Donate a beverage to help me to keep Seq2SeqSharp up to date :) [![Support via PayPal](https://www.paypalobjects.com/en_GB/i/btn/btn_donate_SM.gif)](https://www.paypal.me/fuzhongkai/)

# Seq2SeqSharp  
Seq2SeqSharp is a tensor based fast & flexible encoder-decoder deep neural network framework written by .NET (C#). It can be used for sequence-to-sequence task, sequence-labeling task and sequence-classification task and other NLP tasks. Seq2SeqSharp supports both CPUs and GPUs. It's powered by .NET core, so Seq2SeqSharp can run on both Windows and Linux without any modification and recompilation.  

# Features  
Pure C# framework   
Transformer encoder and decoder  
Attention based LSTM decoder with coverage model  
Bi-directional LSTM encoder  
Support multi-platforms, such as Windows, Linux, Mac and others  
Built-in several networks for sequence-to-sequence task and sequence-labeling task  
Graph based neural network  
Automatic differentiation  
Tensor based operations  
Running on both CPUs and GPUs (CUDA)  
Support Tensor Cores  
Support multi-GPUs  
Optimized CUDA memory management for higher performance  
Mini-batch  
Dropout  
RMSProp and Adam optmization  
Embedding & Pre-trained model 
Built-in metrics and extendable, such as BLEU score, Length ratio, F1 score and so on  
Auto data shuffling  
Auto vocabulary building  
Beam search decoder  
Visualize neural network  

# Architecture  
Here is the architecture of Seq2SeqSharp  
![](https://raw.githubusercontent.com/zhongkaifu/Seq2SeqSharp/master/Overview.jpg)

Seq2SeqSharp provides the unified tensor operations, which means all tensor operations running on CPUs and GPUs are completely same and they can get switched on different device types without any modification.  
Seq2SeqSharp is also a framework that neural networks can run on multi-GPUs in parallel. It can automatically distribute/sync weights/gradients over devices, manage resources and models and so on, so developers are able to totally focus on how to design and implment networks for their tasks.  
Seq2SeqSharp is built by (.NET core)[https://docs.microsoft.com/en-us/dotnet/core/], so it can run on both Windows and Linux without any modification and recompilation.  

# Usage  
Seq2SeqSharp provides two console tools that you can run for sequence-to-sequence task (**Seq2SeqConsole.exe**) and sequence-labeling task (**SeqLabelConsole.exe**).  

You can use Seq2SeqConsole tool to train, test and visualize models.  
Here is the command line to train a model:  
**Seq2SeqConsole.exe -TaskName Train [parameters...]**  
Parameters:  
**-SrcEmbeddingDim**: The embedding dim of source side. Default is 128  
**-TgtEmbeddingDim**: The embedding dim of target side. Default is 128  
**-HiddenSize**: The hidden layer dim of encoder and decoder.  Default is 128    
**-LearningRate**: Learning rate. Default is 0.001  
**-EncoderLayerDepth**: The network depth in encoder. The default depth is 1.  
**-DecoderLayerDepth**: The network depth in decoder. The default depth is 1.  
**-EncoderType**: The type of encoder. It supports BiLSTM and Transformer.  
**-DecoderType**: The type of decoder. It supports AttentionLSTM and Transformer.  
**-MultiHeadNum**: The number of multi-heads in Transformer encoder and decoder.  
**-ModelFilePath**: The model file path for training and testing.  
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
**-BatchSize**: Batch size for training. Default is 1.  
**-ValBatchSize**: Batch size for testing. Default is 1.  
**-Dropout**: Dropout ratio. Defaul is 0.1  
**-ProcessorType**: Processor type: CPU or GPU(Cuda)  
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-MaxEpochNum**: Maxmium epoch number during training. Default is 100  
**-MaxSrcSentLength**: Maxmium sentence length on source side. Default is 32 tokens  
**-MaxTgtSentLength**: Maxmium sentence length on target side. Default is 32 tokens  
**-WarmUpSteps**: The number of steps for warming up. Default is 8,000  
**-CompilerOptions**: The options for CUDA NVRTC compiler. Options are split by space. For example: "--use_fast_math --gpu-architecture=compute_60" means to use fast math libs and run on Pascal and above GPUs  
**-Optimizer**: The weights optimizer during training. It supports Adam and RMSProp. Adam is default  

Note that:  
  1) if "-SrcVocab" and "-TgtVocab" are empty, vocabulary will be built from training corpus.  
  2) Txt2Vec for external embedding model building can get downloaded from https://github.com/zhongkaifu/Txt2Vec  

Example: Seq2SeqConsole.exe -TaskName Train -SrcEmbeddingDim 512 -TgtEmbeddingDim 512 -HiddenSize 512 -LearningRate 0.002 -ModelFilePath seq2seq.model -TrainCorpusPath .\corpus -ValidCorpusPath .\corpus_valid -SrcLang ENU -TgtLang CHS -BatchSize 256 -ProcessorType GPU -EncoderType Transformer -EncoderLayerDepth 6 -DecoderLayerDepth 2 -MultiHeadNum 8 -DeviceIds 0,1,2,3,4,5,6,7  

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
**-OutputFile**: The test result file.  
**-ModelFilePath**: The trained model file path. 
**-ProcessorType**: Architecture type: CPU or GPU 
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-BeamSearchSize**: Beam search size. Default is 1  
**-MaxSrcSentLength**: Maxmium sentence length on source side. Default is 32 tokens  
**-MaxTgtSentLength**: Maxmium sentence length on target side. Default is 32 tokens  

Example: Seq2SeqConsole.exe -TaskName Test -ModelFilePath seq2seq.model -InputTestFile test.txt -OutputFile result.txt -ProcessorType CPU -BeamSearchSize 5 -MaxSrcSentLength 100 -MaxTgtSentLength 100  

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
        "BatchSize":256,
        "BeamSearchSize":1,
        "Beta1":0.9,
        "Beta2":0.98,
        "CompilerOptions":"--use_fast_math --gpu-architecture=compute_70",
        "DecoderLayerDepth":6,
        "DecoderType":"Transformer",
        "DeviceIds":"0,1,2,3",
        "DropoutRatio":0.0,
        "EnableCoverageModel":false,
        "EnableSegmentEmbeddings":false,
        "EncoderLayerDepth":6,
        "EncoderType":"Transformer",
        "HiddenSize":512,
        "InputTestFile":null,
        "GradClip":5.0,
        "MaxEpochNum":100,
        "MaxSrcSentLength":100,
        "MaxTgtSentLength":100,
        "ModelFilePath":"seq2seq.model",
        "MultiHeadNum":8,
        "Optimizer":"Adam",
        "OutputFile":null,
        "ProcessorType":"GPU",   
        "SharedEmbeddings":false,     
        "ShuffleBlockSize":-1,
        "ShuffleType": "NoPaddingInSrc",
        "SrcEmbeddingDim":512,
        "SrcEmbeddingModelFilePath":null,
        "SrcLang":"ENU",
        "SrcVocab":null,
        "StartLearningRate":0.0004,
        "TaskName":"Train",        
        "TgtEmbeddingDim":512,    
        "TgtEmbeddingModelFilePath":null,
        "TgtLang":"CHS",
        "WeightsUpdateCount":0,                
        "TgtVocab":null,
        "TrainCorpusPath":"corpus",
        "ValidCorpusPath":"corpus_valid",                
        "ValBatchSize":16,                                   
        "WarmUpSteps":8000,
        "VisualizeNNFilePath":null            
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

# Release Package  
You can download the release package from (here)[https://github.com/zhongkaifu/Seq2SeqSharp/releases/tag/20210125] . The release package includes Seq2SeqSharp binary files, model file and test files. The model was trained for English to Chinese translation. It's an encoder-decoder model using 4 Transformer layers (hidden dim: 512, embedding dim: 512). The training config file is also included in the package. Test input file contains one English sentence per line, and the test reference file has one Chinese sentence per line. All sentences were already encoded to subwords by SentencePiece, so the package also includes the model and vocabulary of SentencePiece.  


(SentencePiece)[https://github.com/google/sentencepiece] is a subword level tokenization tool from Google. Given raw text, it can build model and vocabulary at subword level, encode text from word level to subword level or decode subword text back to word level. Subword level tokenization could significantly reduce vocabulary size which is useful for OOV and decoding performance improvement for many systems, especially low resource systems. Subword level tokenization and SentencePiece are optional for Seq2SeqSharp. You can tokenize input text to any type of tokens and send them to Seq2SeqSharp or let Seq2SeqSharp generate them.  


The model in the release package was trained by training corpus processed by SentencePiece, so inputs and outputs text of this model needs to be pre-processed by SentencePiece. Again, you could train your model with/without SentencePiece. It's totally optional.  


Here are steps on how to play it.  

0. Preparation  

   0.1 Install Nvidia driver and Cuda 11  

      Windows: Download (Nvidia driver)[https://www.nvidia.com/Download/index.aspx] and (Cuda 11)[https://developer.nvidia.com/cuda-11.1.0-download-archive], and then install them.  

      Linux: You can use apt to update drivers and cuda, for example: sudo apt install nvidia-driver-440  

   0.2 Install dotNet core  

      Windows: Download (.NET Core)[https://docs.microsoft.com/en-us/dotnet/core/] and install.  

      Linux: You can use the following apt-get commands to download and install it:  

         sudo apt-get update  

         sudo apt-get install -y apt-transport-https  

         sudo apt-get update  

         sudo apt-get install -y aspnetcore-runtime-5.0  


   0.3 Install SentencePiece (optional)  

      You can follow instructions on (SentencePiece github)[https://github.com/google/sentencepiece] to download and install it. It supports both Windows and Linux.  


1. Run SentencePiece to encode raw input English text to subword (optional)  

   You can run the following example command for encoding: spm_encode --model=enuSpm.model test_raw.txt > test_spm.txt  

   The test input files in the release package are already encoded, so you do not have to do it.   



2. Run Seq2SeqSharp to translate the above input text from English to Chinese  

   You can run the following command for translation.  

      Seq2SeqConsole.exe -TaskName Test -ModelFilePath seq2seq_mt.model -InputTestFile test_spm.txt -OutputTestFile out.txt -MaxSrcSentLength 100 -MaxTgtSentLength 100 -ProcessorType CPU  


3. Run SentencePiece to decode output Chinese text (optional)  

   You can run the following command for decoding: spm_decode --model=chsSpm.model out.txt > out_raw.txt    


4. Check quality by comparing output Chinese text with reference text   

# Build From Source Code  
Besides using the release package, you could also build Seq2SeqSharp from source code. It has just two steps:  

1. Clone the project from github: git clone https://github.com/zhongkaifu/Seq2SeqSharp.git  
2. Build all projects: dotnet build Seq2SeqSharp.sln --configuration Release  

# Using different CUDA versions and .NET versions  
Seq2SeqSharp uses CUDA 11.x and .NET 5.0 by default, but you can still use different versions of them. It has already been tested on .NET core 3.1, CUDA 10.x and some other versions.  

For different CUDA versions, you need to change the versions of ManagedCUDA to the corresponding versions. They are all in *.project files. For example: The following settings are in TensorSharp.CUDA.project for CUDA 10.2  
```xml
    <PackageReference Include="ManagedCuda-102">  
      <Version>10.2.41</Version>  
    </PackageReference>  
    <PackageReference Include="ManagedCuda-CUBLAS">  
      <Version>10.2.41</Version>  
    </PackageReference>  
    <PackageReference Include="ManagedCuda-NVRTC">  
      <Version>10.2.41</Version>  
    </PackageReference>  
```

For different .NET versions, you need to modify target framework in *.csproj files. Here is an example to use .net core 3.1 as target framework in Seq2SeqSharp.csproj file.  
```xml
    <PropertyGroup>  
      <TargetFramework>netcoreapp3.1</TargetFramework>  
    </PropertyGroup>  
```

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
        /// <param name="inputQ">The input Q tensor</param>
        /// <param name="inputK">The input K tensor</param>
        /// <param name="inputV">The input V tensor</param>
        /// <param name="keyMask">The mask for softmax</param>
        /// <param name="batchSize">Batch size of input data set</param>
        /// <param name="graph">The instance of computing graph</param>
        /// <returns>Transformered output tensor</returns>
        public IWeightTensor Perform(IWeightTensor inputQ, IWeightTensor inputK, IWeightTensor inputV, IWeightTensor keyMask, int batchSize, IComputeGraph graph)
        {
            using (IComputeGraph g = graph.CreateSubGraph($"{m_name}_MultiHeadAttention"))
            {
                int seqLenQ = inputQ.Rows / batchSize;

                // SeqLenK must be euqal to SeqLenV
                int seqLenK = inputK.Rows / batchSize;
                int seqLenV = inputV.Rows / batchSize;

                IWeightTensor inputQNorm = layerNormQ.Norm(inputQ, g);
                if (inputK == inputQ)
                {
                    inputK = inputQNorm;
                }
                if (inputV == inputQ)
                {
                    inputV = inputQNorm;
                }

                //Input projections
                float scale = 1.0f;
                IWeightTensor allQ = g.View(g.Affine(inputQNorm, Q, Qb, scale), dims: new long[] { batchSize, seqLenQ, m_multiHeadNum, m_d });
                IWeightTensor allK = g.View(g.Affine(inputK, K, Kb, scale), dims: new long[] { batchSize, seqLenK, m_multiHeadNum, m_d });
                IWeightTensor allV = g.View(g.Affine(inputV, V, Vb, scale), dims: new long[] { batchSize, seqLenV, m_multiHeadNum, m_d });

                //Multi-head attentions
                IWeightTensor Qs = g.View(g.AsContiguous(g.Transpose(allQ, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor Ks = g.View(g.AsContiguous(g.Transpose(g.Transpose(allK, 1, 2), 2, 3)), dims: new long[] { batchSize * m_multiHeadNum, m_d, seqLenK });
                IWeightTensor Vs = g.View(g.AsContiguous(g.Transpose(allV, 1, 2)), dims: new long[] { batchSize * m_multiHeadNum, seqLenV, m_d });

                // Scaled softmax
                scale = 1.0f / (float)(Math.Sqrt(m_d));
                IWeightTensor attn = g.MulBatch(Qs, Ks, batchSize * m_multiHeadNum, scale);

                if (keyMask != null)
                {
                    using (var keyMaskView = g.View(keyMask, runGradient: false, dims: new long[] { batchSize, 1, seqLenQ, seqLenK }))
                    {
                        using (var keyMaskViewExp = g.Expand(keyMaskView, runGradient: false, dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, seqLenK }))
                        {
                            using (var keyMaskViewExpConti = g.AsContiguous(keyMaskViewExp, runGradient: false))
                            {
                                using (var keyMaskViewExpContiView = g.View(keyMaskViewExpConti, runGradient: false, dims: new long[] { batchSize * m_multiHeadNum, seqLenQ, seqLenK }))
                                {
                                    attn = g.Add(attn, keyMaskViewExpContiView, runGradient1: true, runGradient2: false);
                                }
                            }
                        }
                    }
                }

                IWeightTensor softmax = g.Softmax(attn, inPlace: true);

                IWeightTensor o = g.View(g.MulBatch(softmax, Vs, batchSize * m_multiHeadNum), dims: new long[] { batchSize, m_multiHeadNum, seqLenQ, m_d });
                IWeightTensor W = g.View(g.AsContiguous(g.Transpose(o, 1, 2)), dims: new long[] { batchSize * seqLenQ, m_multiHeadNum * m_d });

                // Output projection
                IWeightTensor finalAttResults = g.Dropout(g.Affine(W, W0, b0), batchSize, m_dropoutRatio, inPlace: true);

                return graph.Add(finalAttResults, inputQ);
            }
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
        ParallelCorpus trainCorpus = new ParallelCorpus(corpusFilePath: opts.TrainCorpusPath, srcLangName: opts.SrcLang, tgtLangName: opts.TgtLang, batchSize: opts.BatchSize, shuffleBlockSize: opts.ShuffleBlockSize,
            maxSrcSentLength: opts.MaxSrcSentLength, maxTgtSentLength: opts.MaxTgtSentLength, shuffleEnums: shuffleType);
        // Load valid corpus
        ParallelCorpus validCorpus = string.IsNullOrEmpty(opts.ValidCorpusPath) ? null : new ParallelCorpus(opts.ValidCorpusPath, opts.SrcLang, opts.TgtLang, opts.ValBatchSize, opts.ShuffleBlockSize, opts.MaxSrcSentLength, opts.MaxTgtSentLength, shuffleEnums: shuffleType);

        // Create learning rate
        ILearningRate learningRate = new DecayLearningRate(opts.StartLearningRate, opts.WarmUpSteps, opts.WeightsUpdateCount);

        // Create optimizer
        IOptimizer optimizer = null;
        if (String.Equals(opts.Optimizer, "Adam", StringComparison.InvariantCultureIgnoreCase))
        {
            optimizer = new AdamOptimizer(opts.GradClip, opts.Beta1, opts.Beta2);
        }
        else
        {
            optimizer = new RMSPropOptimizer(opts.GradClip, opts.Beta1);
        }

        // Create metrics
        List<IMetric> metrics = new List<IMetric>
        {
            new BleuMetric(),
            new LengthRatioMetric()
        };

        if (!String.IsNullOrEmpty(opts.ModelFilePath) && File.Exists(opts.ModelFilePath))
        {
            //Incremental training
            Logger.WriteLine($"Loading model from '{opts.ModelFilePath}'...");
            ss = new AttentionSeq2Seq(modelFilePath: opts.ModelFilePath, processorType: processorType, dropoutRatio: opts.DropoutRatio, deviceIds: deviceIds,
            isSrcEmbTrainable: opts.IsSrcEmbeddingTrainable, isTgtEmbTrainable: opts.IsTgtEmbeddingTrainable, isEncoderTrainable: opts.IsEncoderTrainable, isDecoderTrainable: opts.IsDecoderTrainable,
            maxSrcSntSize: opts.MaxSrcSentLength, maxTgtSntSize: opts.MaxTgtSentLength, memoryUsageRatio: opts.MemoryUsageRatio, shuffleType: shuffleType, compilerOptions: cudaCompilerOptions);
        }
        else
        {
            // Load or build vocabulary
            Vocab vocab = null;
            if (!string.IsNullOrEmpty(opts.SrcVocab) && !string.IsNullOrEmpty(opts.TgtVocab))
            {
                // Vocabulary files are specified, so we load them
                vocab = new Vocab(opts.SrcVocab, opts.TgtVocab);
            }
            else
            {
                // We don't specify vocabulary, so we build it from train corpus
                vocab = new Vocab(trainCorpus);
            }

            //New training
            ss = new AttentionSeq2Seq(srcEmbeddingDim: opts.SrcEmbeddingDim, tgtEmbeddingDim: opts.TgtEmbeddingDim, hiddenDim: opts.HiddenSize, encoderLayerDepth: opts.EncoderLayerDepth, decoderLayerDepth: opts.DecoderLayerDepth,
                srcEmbeddingFilePath: opts.SrcEmbeddingModelFilePath, tgtEmbeddingFilePath: opts.TgtEmbeddingModelFilePath, vocab: vocab, modelFilePath: opts.ModelFilePath,
                dropoutRatio: opts.DropoutRatio, processorType: processorType, deviceIds: deviceIds, multiHeadNum: opts.MultiHeadNum, encoderType: encoderType, decoderType: decoderType,
                maxSrcSntSize: opts.MaxSrcSentLength, maxTgtSntSize: opts.MaxTgtSentLength, enableCoverageModel: opts.EnableCoverageModel, memoryUsageRatio: opts.MemoryUsageRatio, shuffleType: shuffleType, compilerOptions: cudaCompilerOptions);
        }

        // Add event handler for monitoring
        ss.IterationDone += ss_IterationDone;

        // Kick off training
        ss.Train(maxTrainingEpoch: opts.MaxEpochNum, trainCorpus: trainCorpus, validCorpus: validCorpus, learningRate: learningRate, optimizer: optimizer, metrics: metrics);
```

# Todo List  
If you are interested in below items, please let me know. Becuase African proverb says "If you want to go fast, go alone. If you want to go far, go together" :)  
And More...  