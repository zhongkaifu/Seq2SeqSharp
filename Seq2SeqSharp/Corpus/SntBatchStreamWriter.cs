using AdvUtils;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Corpus
{
    public class SntBatchStreamWriter
    {
        static object locker = new object();
        private string filePath;

        private List<string> outputBuffer = new List<string>();
        StreamWriter sw;
        int endWriteIdx = 0;
        int numWrittenToFile = 0;

        SentencePiece sp;

        public SntBatchStreamWriter(string filePath, string sentencePieceModelPath = null)
        {
            this.filePath = filePath;
            sw = new StreamWriter(filePath, false);

            if (String.IsNullOrEmpty(sentencePieceModelPath) == false)
            {
                Logger.WriteLine($"Loading sentence piece model '{sentencePieceModelPath}' for decoding.");
                sp = new SentencePiece(sentencePieceModelPath);
            }

        }

        public void WriteResults(int idx, List<NetworkResult> results)
        {
            lock (locker)
            {
                List<string> finalOutputs = new List<string>();

                List<List<string>> outputAllTasks = new List<List<string>>();
                foreach (NetworkResult result in results) //for result in each task, shape [beam size, batch size, tgt token size]
                {
                    List<string> outputEntireBatch = new List<string>();
                    for (int batchIdx = 0; batchIdx < result.Output[0].Count; batchIdx++)
                    {
                        List<string> outputBeams = new List<string>();
                        for (int beamIdx = 0; beamIdx < result.Output.Count; beamIdx++)
                        {
                            string output = string.Join(" ", result.Output[beamIdx][batchIdx]);
                            outputBeams.Add(output);
                        }

                        string outputAllBeams = string.Join("\t", outputBeams);
                        outputEntireBatch.Add(outputAllBeams);
                    }

                    outputAllTasks.Add(outputEntireBatch);
                }

                int batchSize = outputAllTasks[0].Count;
                for (int i = 0; i < batchSize; i++) // iterate all output line
                {
                    List<string> output = new List<string>();
                    for (int j = 0; j < outputAllTasks.Count; j++) // iterate all tasks result
                    {
                        output.Add(outputAllTasks[j][i]);
                    }

                    finalOutputs.Add(string.Join("\t", output));
                }

                //Need to extend output buffer
                while (outputBuffer.Count < idx + batchSize - numWrittenToFile)
                {
                    outputBuffer.Add(null);
                }

                for (int i = 0; i < batchSize; i++)
                {
                    outputBuffer[idx + i - numWrittenToFile] = finalOutputs[i];
                }



                while(endWriteIdx < outputBuffer.Count)
                {
                    if (outputBuffer[endWriteIdx] == null)
                    {
                        break;
                    }

                    string line = outputBuffer[endWriteIdx];
                    if (sp != null)
                    {
                        line = sp.Decode(line);
                    }

                    sw.WriteLine(line);

                    endWriteIdx++;
                }

                if (endWriteIdx > 0)
                {
                    outputBuffer.RemoveRange(0, endWriteIdx);
                    numWrittenToFile += endWriteIdx;
                    endWriteIdx = 0;
                }

            }
        }

        public void Close()
        {
            sw.Close();
        }
    }
}
