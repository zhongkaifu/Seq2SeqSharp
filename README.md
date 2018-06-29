# Seq2SeqSharp
========
Seq2SeqSharp is an encoder-decoder deep neural network framework based on .NET Framework. Encoder is based on bidirectional LSTM neural network, and decoder is based on LSTM-Attention neural network. It supports both dense and sparse feature types.

# Usage
========
You could use Seq2SeqConsole tool to train and test models.  

Here is the command line to train a model:
**Seq2SeqConsole.exe train [parameters...]**  
Parameters:  
**-WordVectorSize**: The vector size of encoded source word.  
**-HiddenSize**: The hidden layer size of encoder. The hidden layer size of decoder is 2x (-HiddenSize)  
**-LearningRate**: Learning rate. Default value is 0.001  
**-Depth**: The network depth in decoder. Default value is 1  
**-ModelFilePath**: The trained model file path.  
**-SrcVocab**: The vocabulary file path for source side.  
**-TgtVocab**: The vocabulary file path for target side.  
**-SrcLang**: Source language name.  
**-TgtLang**: Target language name.  
**-TrainCorpusPath**: training corpus folder path  
**-UseSparseFeature**: It indicates if sparse feature used for training.  

Here is the command line to test models  
**Seq2SeqConsole.exe predict [parameters...]**  
Parameters:  
**-InputTestFile**: The input file for test.  
**-OutputTestFile**: The test result file.  
**-ModelFilePath**: The trained model file path.  


# Performance
========
Seq2SeqSharp leverages System.Numerics.Vector to get better performance by SSE/AVX/AVX2 instructions. If you have Intel CPUs, you could download Intel MKL(https://software.intel.com/en-us/mkl) add "MKL" into Seq2SeqSharp project's conditioanl compilation symbols to get much higher performance by Intel MKL.  
