// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using Seq2SeqSharp._SentencePiece;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.IO;

namespace Seq2SeqSharp.Corpus
{
    public class SntBatchStreamWriter
    {
        static object locker = new object();
        private List<string> outputBuffer = new List<string>();
        private List<string> alignmentBuffer = new List<string>();
        StreamWriter sw = null;
        StreamWriter swAlignment = null;
        int endWriteIdx = 0;
        int numWrittenToFile = 0;

        SentencePiece sp;

        public SntBatchStreamWriter(string filePath = null, string sentencePieceModelPath = null, string alignmentFilePath = null)
        {
            if (String.IsNullOrEmpty(filePath) == false)
            {
                sw = new StreamWriter(filePath, false);
            }

            if (String.IsNullOrEmpty(alignmentFilePath) == false)
            {
                swAlignment = new StreamWriter(alignmentFilePath, false);
            }

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
                List<string> finalOutputs = null;
                int batchSize = results[0].Output[0].Count;
                if (sw != null)
                {
                    finalOutputs = new List<string>();
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

                    for (int i = 0; i < batchSize; i++) // iterate all output line
                    {
                        List<string> output = new List<string>();
                        for (int j = 0; j < outputAllTasks.Count; j++) // iterate all tasks result
                        {
                            output.Add(outputAllTasks[j][i]);
                        }

                        finalOutputs.Add(string.Join("\t", output));
                    }
                }

                List<string> finalAlignments = null;
                if (swAlignment != null)
                {
                    finalAlignments = new List<string>();
                    List<List<string>> alignmentAllTasks = new List<List<string>>();
                    foreach (NetworkResult result in results) //for result in each task, shape [beam size, batch size, tgt token size]
                    {
                        List<string> alignmentEntireBatch = new List<string>();
                        for (int batchIdx = 0; batchIdx < result.Alignments[0].Count; batchIdx++)
                        {
                            List<string> alignmentBeams = new List<string>();
                            for (int beamIdx = 0; beamIdx < result.Alignments.Count; beamIdx++)
                            {
                                List<string> alignmentItems = new List<string>();
                                for (int k = 0; k < result.Alignments[beamIdx][batchIdx].Count; k++)
                                {
                                    alignmentItems.Add($"{result.Alignments[beamIdx][batchIdx][k]}({result.AlignmentScores[beamIdx][batchIdx][k].ToString("F2")})");
                                }
                                string alignment = string.Join(" ", alignmentItems);
                                alignmentBeams.Add(alignment);
                            }

                            string alignmentAllBeams = string.Join("\t", alignmentBeams);
                            alignmentEntireBatch.Add(alignmentAllBeams);
                        }

                        alignmentAllTasks.Add(alignmentEntireBatch);
                    }

                    for (int i = 0; i < batchSize; i++) // iterate all output line
                    {
                        List<string> alignment = new List<string>();
                        for (int j = 0; j < alignmentAllTasks.Count; j++) // iterate all tasks result
                        {
                            alignment.Add(alignmentAllTasks[j][i]);
                        }

                        finalAlignments.Add(string.Join("\t", alignment));
                    }
                }

                int buffSize = 0;
                if (sw != null)
                {
                    //Need to extend output buffer
                    while (outputBuffer.Count < idx + batchSize - numWrittenToFile)
                    {
                        outputBuffer.Add(null);
                    }

                    for (int i = 0; i < batchSize; i++)
                    {
                        outputBuffer[idx + i - numWrittenToFile] = finalOutputs[i];
                    }
                    buffSize = outputBuffer.Count;
                }

                if (swAlignment != null)
                {
                    //Need to extend output buffer
                    while (alignmentBuffer.Count < idx + batchSize - numWrittenToFile)
                    {
                        alignmentBuffer.Add(null);
                    }

                    for (int i = 0; i < batchSize; i++)
                    {
                        alignmentBuffer[idx + i - numWrittenToFile] = finalAlignments[i];
                    }
                    buffSize = alignmentBuffer.Count;
                }


                while (endWriteIdx < buffSize)
                {
                    if (sw != null)
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
                    }

                    if (swAlignment != null)
                    {
                        if (alignmentBuffer[endWriteIdx] == null)
                        {
                            break;
                        }

                        swAlignment.WriteLine(alignmentBuffer[endWriteIdx]);
                    }

                    endWriteIdx++;
                }

                if (endWriteIdx > 0)
                {
                    if (sw != null)
                    {
                        outputBuffer.RemoveRange(0, endWriteIdx);
                    }

                    if (swAlignment != null)
                    {
                        alignmentBuffer.RemoveRange(0, endWriteIdx);
                    }

                    numWrittenToFile += endWriteIdx;
                    endWriteIdx = 0;
                }

            }
        }

        public void Close()
        {
            if (sw != null)
            {
                sw.Close();
            }

            if (swAlignment != null)
            {
                swAlignment.Close();
            }
        }
    }
}
