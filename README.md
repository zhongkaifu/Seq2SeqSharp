# Seq2SeqSharp  
Seq2SeqSharp is an C# encoder-decoder deep neural network framework running on both CPU and GPU  

# Features  
Pure C# framework   
Deep bi-directional LSTM encoder  
Deep attention based LSTM decoder  
Transformer encoder  
Graph based neural network  
Automatic differentiation  
Tensor based operations  
Running on both CPU and GPU (CUDA)  
Support multi-GPUs  
Mini-batch  
Dropout  
RMSProp optmization  
Pre-trained model  
Auto data shuffling  
Auto vocabulary building  
Beam search decoder  

# Usage  
You could use Seq2SeqConsole tool to train and test models.  

Here is the command line to train a model:
**Seq2SeqConsole.exe -TaskName train [parameters...]**  
Parameters:  
**-WordVectorSize**: The vector size of encoded source word.  
**-HiddenSize**: The hidden layer size of encoder and decoder.    
**-LearningRate**: Learning rate. Default is 0.001  
**-Depth**: The network depth in decoder. Default is 1  
**-ModelFilePath**: The trained model file path.  
**-SrcVocab**: The vocabulary file path for source side.  
**-TgtVocab**: The vocabulary file path for target side.  
**-SrcEmbedding**: The external embedding model file path for source side. It is built by Txt2Vec project.  
**-TgtEmbedding**: The external embedding model file path for target side. It is built by Txt2Vec project.  
**-SrcLang**: Source language name.  
**-TgtLang**: Target language name.  
**-TrainCorpusPath**: training corpus folder path  
**-BatchSize**: Mini-batch size. Default is 1. For CPU runner, it must be 1.  
**-DropoutRatio**: Dropout ratio. Defaul is 0.1  
**-ArchType**: Runner type. 0 - GPU (CUDA), 1 - CPU (Intel MKL), 2 - CPU. Default is 0  
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-MaxEpochNum**: Maxmium epoch number during training. Default is 100  
Note that:  
  1) if "-SrcVocab" and "-TgtVocab" are empty, vocabulary will be built from training corpus.  
  2) Txt2Vec for external embedding model building can get downloaded from https://github.com/zhongkaifu/Txt2Vec  

Example: Seq2SeqConsole.exe -TaskName train -WordVectorSize 1024 -HiddenSize 1024 -LearningRate 0.001 -Depth 2 -TrainCorpusPath .\corpus -ModelFilePath nmt.model -SrcLang enu -TgtLang chs -ArchType 0 -DeviceIds 0,1,2,3  

During training, the iteration information will be printed out and logged as follows:  
info,3/13/2019 5:40:22 AM Epoch = '0' LR = '0.002', Current Cost = '3.213204', Avg Cost = '4.764458', SentInTotal = '17612800', SentPerMin = '44415.11', WordPerSec = '37689.26'  
info,3/13/2019 5:49:16 AM Epoch = '0' LR = '0.002', Current Cost = '3.172645', Avg Cost = '4.731404', SentInTotal = '18022400', SentPerMin = '44451.65', WordPerSec = '37674.58'  

Here is the command line to test models  
**Seq2SeqConsole.exe -TaskName test [parameters...]**  
Parameters:  
**-InputTestFile**: The input file for test.  
**-OutputTestFile**: The test result file.  
**-ModelFilePath**: The trained model file path. 
**-ArchType**: Runner type. 0 - GPU (CUDA), 1 - CPU, 2 - CPU (Intel MKL). Default is 0  
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  
**-BeamSearch**: Beam search size. Default is 1  

Example: Seq2SeqConsole.exe -TaskName test -ModelFilePath seq2seq_256.model -InputTestFile test.txt -OutputTestFile result.txt -ArchType 1 -BeamSearch 5  

# Data Format  
The corpus contains each sentence per line. The file name pattern is "mainfilename.{source language name}.snt" and "mainfilename.{target language name}.snt".    
For example: Let's use three letters name CHS for Chinese and ENU for English in Chinese-English parallel corpus, so we could have these corpus files: train01.enu.snt, train01.chs.snt, train02.enu.snt and train02.chs.snt.  
In train01.enu.snt, assume we have below two sentences:  
the children huddled together for warmth .  
the car business is constantly changing .  
So, train01.chs.snt has the corresponding translated sentences:  
孩子 们 挤 成 一 团 以 取暖 .  
汽车 业 也 在 不断 地 变化 .  

# Build Your Neural Networks  
Benefit from automatic differentiation, tensor based compute graph and other features, you can easily build your neural network by a few code. The only thing you need to implment is forward part, and the framework will automatically build the corresponding backward part for you, and make the network could run on multi-GPUs or CPUs.  
Here is an example about attentioned based LSTM cells.  
```c#
        /// <summary>
        /// Update LSTM-Attention cells according to given weights
        /// </summary>
        /// <param name="context">The context weights for attention</param>
        /// <param name="input">The input weights</param>
        /// <param name="computeGraph">The compute graph to build workflow</param>
        /// <returns>Update hidden weights</returns>
        public IWeightMatrix Step(IWeightMatrix context, IWeightMatrix input, IComputeGraph computeGraph)
        {
            var cell_prev = ct;
            var hidden_prev = ht;

            var hxhc = computeGraph.ConcatColumns(input, hidden_prev, context);
            var bs = computeGraph.RepeatRows(b, input.Rows);
            var hhSum = computeGraph.MulAdd(hxhc, Wxhc, bs);
            var hhSum2 = layerNorm1.Process(hhSum, computeGraph);

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum2, hdim * 3, hdim);
            var gates = computeGraph.Sigmoid(gates_raw);
            var cell_write = computeGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, hdim, hdim, hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            ct = computeGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(ct, computeGraph);

            ht = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct2));

            return ht;
        }
```
Another example about scaled multi-heads attention component which is the core part in Transformer model.  
```c#
        /// <summary>
        /// Scaled multi-heads attention component with skip connectioned feed forward layers
        /// </summary>
        /// <param name="input">The input tensor</param>
        /// <param name="g">The instance of computing graph</param>
        /// <returns></returns>
        public IWeightMatrix Perform(IWeightMatrix input, IComputeGraph g)
        {
            var seqLen = input.Rows / m_batchSize;

            //Input projections
            var allQ = g.View(Q.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);

            var allK = g.View(K.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);

            var allV = g.View(V.Process(input, g), m_batchSize, seqLen, m_multiHeadNum, m_d);

            //Multi-head attentions
            var Qs = g.View(g.Permute(allQ, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);
            var Ks = g.View(g.Permute(allK, 2, 0, 3, 1), m_multiHeadNum * m_batchSize, m_d, seqLen);
            var Vs = g.View(g.Permute(allV, 2, 0, 1, 3), m_multiHeadNum * m_batchSize, seqLen, m_d);

            float scale = 1.0f / (float)Math.Sqrt(m_d);

            var attn = g.MulBatch(Qs, Ks, m_multiHeadNum * m_batchSize, scale);
            var attn2 = g.View(attn, m_multiHeadNum * m_batchSize * seqLen, seqLen);

            var softmax = g.Softmax(attn2);
            var softmax2 = g.View(softmax, m_multiHeadNum * m_batchSize, seqLen, seqLen);
            var o = g.View(g.MulBatch(softmax2, Vs, m_multiHeadNum * m_batchSize), m_multiHeadNum, m_batchSize, seqLen, m_d);
            var W = g.View(g.Permute(o, 1, 2, 0, 3), m_batchSize * seqLen, m_multiHeadNum * m_d);

            // Output projection
            var b0s = g.RepeatRows(b0, W.Rows);
            var finalAttResults = g.MulAdd(W, W0, b0s);

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

# Todo List  
If you are interested in below items, please let me know. Becuase African proverb says "If you want to go fast, go alone. If you want to go far, go together" :)  
Support Tensor Cores in CUDA  
Support Half-Float Type (FP16)  
And More...  