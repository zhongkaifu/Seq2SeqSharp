# Seq2SeqSharp  
Seq2SeqSharp is an C# encoder-decoder deep neural network framework running on both CPU and GPU  

# Features  
Pure C# framework (except kernel C code in CUDA)  
Deep bi-directional LSTM encoder  
Deep attention based LSTM decoder  
Graph based neural network  
Automatic differentiation  
Tensor based operations  
Running on both CPU (Intel MKL lib) and GPU (CUDA)  
Support multi-GPUs  
Mini-batch  
Dropout  
RMSProp optmization  
Pre-trained model  
Auto data shuffling  
Auto vocabulary building  

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
Note that:  
  1) if "-SrcVocab" and "-TgtVocab" are empty, vocabulary will be built from training corpus.  
  2) Txt2Vec for external embedding model building can get downloaded from https://github.com/zhongkaifu/Txt2Vec  

Example: Seq2SeqConsole.exe -TaskName train -WordVectorSize 1024 -HiddenSize 1024 -LearningRate 0.001 -Depth 2 -TrainCorpusPath .\corpus -ModelFilePath nmt.model -SrcLang enu -TgtLang chs -ArchType 0 -DeviceIds 0,1,2,3  

Here is a snapshot while Seq2SeqSharp is encoding a new model.  
![](https://github.com/zhongkaifu/Seq2SeqSharp/blob/master/Seq2SeqSharp_Snapshot.JPG)

Here is the command line to test models  
**Seq2SeqConsole.exe -TaskName test [parameters...]**  
Parameters:  
**-InputTestFile**: The input file for test.  
**-OutputTestFile**: The test result file.  
**-ModelFilePath**: The trained model file path. 
**-ArchType**: Runner type. 0 - GPU (CUDA), 1 - CPU (Intel MKL), 2 - CPU. Default is 0  
**-DeviceIds**: Device ids for training in GPU mode. Default is 0. For multi devices, ids are split by comma, for example: 0,1,2  

Example: Seq2SeqConsole.exe -TaskName test -ModelFilePath seq2seq_256.model -InputTestFile test.txt -OutputTestFile result.txt -ArchType 2  

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
Benefit from automatic differentiation, tensor based compute graph and other features, you can easily build your neural network by a few of code. The only thing you need to implment is forward part, and the framework will automatically build the corresponding backward part for you, and make the network could run on multi-GPUs or CPUs.  
Here is an example for attentioned based LSTM.  
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

            (var gates_raw, var cell_write_raw) = computeGraph.SplitColumns(hhSum, hdim * 3, hdim);
            var gates = computeGraph.Sigmoid(gates_raw);
            var cell_write = computeGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = computeGraph.SplitColumns(gates, hdim, hdim, hdim);

            // compute new cell activation
            var retain_cell = computeGraph.EltMul(forget_gate, cell_prev);
            var write_cell = computeGraph.EltMul(input_gate, cell_write);

            ct = computeGraph.Add(retain_cell, write_cell);
            ht = computeGraph.EltMul(output_gate, computeGraph.Tanh(ct));

            return ht;
        }
```
