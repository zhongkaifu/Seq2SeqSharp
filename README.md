# Seq2SeqSharp  
Seq2SeqSharp is an C# encoder-decoder deep neural network framework. The encoder is bidirectional LSTM neural network, and the decoder is LSTM-Attention neural network. It has automatic differentiation feature, supports both dense and sparse feature types, and could run on both CPU and GPU mode.

# Usage  
You could use Seq2SeqConsole tool to train and test models.  

Here is the command line to train a model:
**Seq2SeqConsole.exe -TaskName train [parameters...]**  
Parameters:  
**-WordVectorSize**: The vector size of encoded source word.  
**-HiddenSize**: The hidden layer size of encoder and decoder.    
**-LearningRate**: Learning rate. Default value is 0.001  
**-Depth**: The network depth in decoder. Default value is 1  
**-ModelFilePath**: The trained model file path.  
**-SrcVocab**: The vocabulary file path for source side.  
**-TgtVocab**: The vocabulary file path for target side.  
**-SrcEmbedding**: The external embedding model file path for source side.  
**-TgtEmbedding**: The external embedding model file path for target side.  
**-SrcLang**: Source language name.  
**-TgtLang**: Target language name.  
**-TrainCorpusPath**: training corpus folder path  
**-SparseFeature**: It indicates if sparse feature used for training. False by default.  
Note that if "-SrcVocab" and "-TgtVocab" are empty, vocabulary will be built from training corpus.  

Example: Seq2SeqConsole.exe -TaskName train -WordVectorSize 128 -HiddenSize 128 -LearningRate 0.001 -Depth 1 -TrainCorpusPath .\corpus -ModelFilePath nmt.model -SrcLang enu -TgtLang chs  

Here is the command line to test models  
**Seq2SeqConsole.exe -TaskName test [parameters...]**  
Parameters:  
**-InputTestFile**: The input file for test.  
**-OutputTestFile**: The test result file.  
**-ModelFilePath**: The trained model file path.  

# Data Format  
The corpus contains each sentence per line. The file name pattern is "mainfilename.{source language name}.snt" and "mainfilename.{target language name}.snt".    
For example: Let's use three letters name CHS for Chinese and ENU for English in Chinese-English parallel corpus, so we could have these corpus files: train01.enu.snt, train01.chs.snt, train02.enu.snt and train02.chs.snt.  
In train01.enu.snt, assume we have below two sentences:  
the children huddled together for warmth .  
the car business is constantly changing .  
So, train01.chs.snt has the corresponding translated sentences:  
孩子 们 挤 成 一 团 以 取暖 .  
汽车 业 也 在 不断 地 变化 .  

# Performance  
Seq2SeqSharp supports both CPU and GPU mode. For CPU mode, it leverages System.Numerics.Vector and Intel MKL(https://software.intel.com/en-us/mkl) to get better performance by SSE/AVX/AVX2 instructions. For GPU mode, it calls CUDA APIs to run on GPU which is powered by modified TensorSharp (https://github.com/alex-weaver/TensorSharp).
Seq2SeqSharp can switch CPU and GPU mode by the setting in project's conditional compilation symbols: "MKL" is for CPU mode with MKL, "CUDA" is for GPU mode. Empty symbol means CPU mode with System.Numerics.Vector only.
Note that if Seq2SeqSharp is compiled with "MKL" compilation symbols, "mklvars.bat" is required to run before launching Seq2SeqSharp.exe, such as "C:\Program Files (x86)\IntelSWTools\parallel_studio_xe_2018\compilers_and_libraries_2018\windows\mkl\bin\mklvars.bat intel64"  
