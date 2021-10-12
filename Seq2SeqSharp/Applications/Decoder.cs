using Seq2SeqSharp.Corpus;
using Seq2SeqSharp.Layers;
using Seq2SeqSharp.Tools;
using Seq2SeqSharp.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Applications
{
    public class Decoder
    {
        public static MultiProcessorNetworkWrapper<IDecoder> CreateDecoders(IModel modelMetaData, Seq2SeqOptions options, RoundArray<int> raDeviceIds, int contextDim)
        {
            MultiProcessorNetworkWrapper<IDecoder> decoder;
            if (modelMetaData.DecoderType == DecoderTypeEnums.AttentionLSTM)
            {
                decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                     new AttentionDecoder("AttnLSTMDecoder", modelMetaData.HiddenDim, modelMetaData.DecoderEmbeddingDim, contextDim,
                     options.DropoutRatio, modelMetaData.DecoderLayerDepth, raDeviceIds.GetNextItem(), modelMetaData.EnableCoverageModel, isTrainable: options.IsDecoderTrainable), raDeviceIds.ToArray());
            }
            else
            {
                decoder = new MultiProcessorNetworkWrapper<IDecoder>(
                    new TransformerDecoder("TransformerDecoder", modelMetaData.MultiHeadNum, modelMetaData.HiddenDim, modelMetaData.DecoderEmbeddingDim, modelMetaData.DecoderLayerDepth, options.DropoutRatio, raDeviceIds.GetNextItem(),
                    isTrainable: options.IsDecoderTrainable, learningRateFactor: options.DecoderStartLearningRateFactor), raDeviceIds.ToArray());
            }

            return decoder;
        }



        public static List<List<int>> ExtractBatchTokens(List<BeamSearchStatus> batchStatus)
        {
            List<List<int>> batchTokens = new List<List<int>>();
            foreach (var item in batchStatus)
            {
                batchTokens.Add(item.OutputIds);
            }

            return batchTokens;
        }

        /// <summary>
        /// Initialize beam search status for all beam search and batch results
        /// </summary>
        /// <param name="beamSearchSize"></param>
        /// <param name="batchSize"></param>
        /// <param name="tgtTokensList"></param>
        /// <returns>shape: (beam_search_size, batch_size)</returns>
        public static List<List<BeamSearchStatus>> InitBeamSearchStatusListList(int batchSize, List<List<int>> tgtTokensList)
        {
            List<List<BeamSearchStatus>> beam2batchStatus = new List<List<BeamSearchStatus>>();

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();
            for (int i = 0; i < batchSize; i++)
            {
                BeamSearchStatus bss = new BeamSearchStatus();
                bss.OutputIds.AddRange(tgtTokensList[i]);
                bssList.Add(bss);
            }

            beam2batchStatus.Add(bssList);

            return beam2batchStatus;
        }



        /// <summary>
        /// swap shape: (beam_search_size, batch_size) <-> (batch_size, beam_search_size)
        /// </summary>
        /// <param name="input"></param>
        public static List<List<BeamSearchStatus>> SwapBeamAndBatch(List<List<BeamSearchStatus>> input)
        {
            int size1 = input.Count;
            int size2 = input[0].Count;

            List<List<BeamSearchStatus>> output = new List<List<BeamSearchStatus>>();
            for (int k = 0; k < size2; k++)
            {
                output.Add(new List<BeamSearchStatus>());
            }

            for (int j = 0; j < size1; j++)
            {
                for (int k = 0; k < size2; k++)
                {
                    output[k].Add(input[j][k]);
                }
            }

                return output;
        }

        public static bool AreAllSentsCompleted(List<List<BeamSearchStatus>> input)
        {
            foreach (var seqs in input)
            {
                foreach (var seq in seqs)
                {
                    if (seq.OutputIds[^1] != (int)SENTTAGS.END)
                    {
                        return false;
                    }
                }
            }

            return true;
        }



        public static List<List<BeamSearchStatus>> MergeTwoBeamSearchStatus(List<List<BeamSearchStatus>> input1, List<List<BeamSearchStatus>> input2)
        {
            if (input1 == null)
            {
                return input2;
            }

            if (input2 == null)
            {
                return input1;
            }

            List<List<BeamSearchStatus>> output = new List<List<BeamSearchStatus>>();
            for (int i = 0; i < input1.Count; i++)
            {
                output.Add(new List<BeamSearchStatus>());
                output[i].AddRange(input1[i]);
                output[i].AddRange(input2[i]);
            }

            return output;
        }

        public static (float, List<List<BeamSearchStatus>>) DecodeTransformer(List<List<int>> tgtSeqs, IComputeGraph g, IWeightTensor encOutputs, TransformerDecoder decoder, IFeedForwardLayer decoderFFLayer,
            IWeightTensor tgtEmbedding, IWeightTensor posEmbedding, float[] srcOriginalLenghts, Vocab tgtVocab, ShuffleEnums shuffleType, float dropoutRatio, bool isTraining = true, int beamSearchSize = 1, 
            bool outputSentScore = true, List<BeamSearchStatus> previousBeamSearchResults = null, TokenGenerationEnums tokenGenerationEnum = TokenGenerationEnums.ArgMax, float topPValue = 0.9f)
        {
            int eosTokenId = tgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            float cost = 0.0f;

            int batchSize = tgtSeqs.Count;
            var tgtOriginalLengths = BuildInTokens.PadSentences(tgtSeqs, eosTokenId);
            int tgtSeqLen = tgtSeqs[0].Count;
            int srcSeqLen = encOutputs.Rows / batchSize;
            IWeightTensor srcTgtMask = (shuffleType == ShuffleEnums.NoPadding || batchSize == 1) ? null : g.BuildSrcTgtMask(srcSeqLen, tgtSeqLen, tgtOriginalLengths, srcOriginalLenghts);
            if (srcTgtMask != null)
            {
                srcTgtMask = g.View(srcTgtMask, false, new long [] { srcTgtMask.Sizes[0], 1, srcTgtMask.Sizes[1], srcTgtMask.Sizes[2]});
            }

            IWeightTensor tgtSelfTriMask;
            if (shuffleType == ShuffleEnums.NoPadding || shuffleType == ShuffleEnums.NoPaddingInTgt || batchSize == 1)
            {
                tgtSelfTriMask = g.BuildTriMask(tgtSeqLen, batchSize);
                tgtSelfTriMask = g.View(tgtSelfTriMask, false, new long[] { 1, 1, tgtSeqLen, tgtSeqLen });
            }
            else
            {
                tgtSelfTriMask = g.BuildSelfTriMask(tgtSeqLen, tgtOriginalLengths);
                tgtSelfTriMask = g.View(tgtSelfTriMask, false, new long[] { batchSize, 1, tgtSeqLen, tgtSeqLen });
            }

            IWeightTensor inputEmbs = TensorUtils.CreateTokensEmbeddings(tgtSeqs, g, tgtEmbedding, tgtOriginalLengths, null, null, tgtVocab, scaleFactor: (float)Math.Sqrt(tgtEmbedding.Columns));
            inputEmbs = PositionEmbedding.AddPositionEmbedding(g, posEmbedding, batchSize, inputEmbs, dropoutRatio);

            IWeightTensor decOutput;
            IWeightTensor decEncAttnProbs;
            (decOutput, decEncAttnProbs) = decoder.Decode(inputEmbs, encOutputs, tgtSelfTriMask, srcTgtMask, batchSize, g, outputAttnWeights: false);


            if (isTraining == false)
            {
                // For inference, we only process last token of each sequence in order to speed up
                float[] decOutputIdx = new float[batchSize];
                for (int i = 0; i < batchSize; i++)
                {
                    decOutputIdx[i] = tgtSeqLen * (i + 1) - 1;
                }

                decOutput = g.IndexSelect(decOutput, decOutputIdx);
            }



            IWeightTensor ffLayer = decoderFFLayer.Process(decOutput, batchSize, g);


            if (isTraining == false)
            {
                ffLayer = g.Mul(ffLayer, 5.0f);
            }

            IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true);

            if (isTraining)
            {
                var leftShiftTgtSeqs = g.LeftShiftTokens(tgtSeqs, eosTokenId);
                var scatterIdxTensor = g.View(leftShiftTgtSeqs, false, new long[] { tgtSeqs.Count * tgtSeqs[0].Count, 1 });
                var gatherTensor = g.Gather(probs, scatterIdxTensor, 1);

                var rnd = new TensorSharp.RandomGenerator();
                int idx = rnd.NextSeed() % (batchSize * tgtSeqLen);
                float score = gatherTensor.GetWeightAt(new long[] { idx, 0 });
                cost += (float)-Math.Log(score + 1e-10f);

                var lossTensor = g.Add(gatherTensor, -1.0f, false);
                TensorUtils.Scatter(probs, lossTensor, scatterIdxTensor, 1);

                ffLayer.CopyWeightsToGradients(probs);

                return (cost, null);
            }
            else
            {
                // Transformer decoder with beam search at inference time
                List<List<BeamSearchStatus>> bssSeqList = new List<List<BeamSearchStatus>>();
                while (beamSearchSize > 0)
                {
                    // Output "i"th target word
                    using var targetIdxTensor = (tokenGenerationEnum == TokenGenerationEnums.ArgMax) ? g.Argmax(probs, 1) : g.TopPSampleIndice(probs, tgtSeqs, topPValue);
                    IWeightTensor gatherTensor = null;
                    if (outputSentScore)
                    {
                        gatherTensor = g.Gather(probs, targetIdxTensor, 1);
                        gatherTensor = g.Log(gatherTensor);
                    }

                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    List<BeamSearchStatus> outputTgtSeqs = new List<BeamSearchStatus>();
                    for (int i = 0; i < batchSize; i++)
                    {
                        BeamSearchStatus bss = new BeamSearchStatus();
                        bss.OutputIds.AddRange(tgtSeqs[i]);
                        bss.OutputIds.Add((int)(targetIdx[i]));

                        if (outputSentScore)
                        {
                            bss.Score = previousBeamSearchResults[i].Score  + -1.0f * gatherTensor.GetWeightAt(new long[] { i, 0});
                        }
                        outputTgtSeqs.Add(bss);
                    }

                    bssSeqList.Add(outputTgtSeqs);

                    beamSearchSize--;
                    if (beamSearchSize > 0)
                    {
                        TensorUtils.ScatterFill(probs, 0.0f, targetIdxTensor, 1);
                    }
                }
                return (0.0f, bssSeqList);
            }
        }

        /// <summary>
        /// Decode output sentences in training
        /// </summary>
        /// <param name="outputSnts">In training mode, they are golden target sentences, otherwise, they are target sentences generated by the decoder</param>
        /// <param name="g"></param>
        /// <param name="encOutputs"></param>
        /// <param name="decoder"></param>
        /// <param name="decoderFFLayer"></param>
        /// <param name="tgtEmbedding"></param>
        /// <returns></returns>
        static public float DecodeAttentionLSTM(List<List<int>> outputSnts, IComputeGraph g, IWeightTensor encOutputs, AttentionDecoder decoder, IFeedForwardLayer decoderFFLayer, IWeightTensor tgtEmbedding, Vocab tgtVocab, int batchSize, bool isTraining = true)
        {
            int eosTokenId = tgtVocab.GetWordIndex(BuildInTokens.EOS, logUnk: true);
            float cost = 0.0f;
            float[] ix_inputs = new float[batchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = outputSnts[i][0];
            }

            // Initialize variables accoridng to current mode
            var originalOutputLengths = isTraining ? BuildInTokens.PadSentences(outputSnts, eosTokenId) : null;
            int seqLen = isTraining ? outputSnts[0].Count : 64;
            HashSet<int> setEndSentId = isTraining ? null : new HashSet<int>();

            // Pre-process for attention model
            AttentionPreProcessResult attPreProcessResult = decoder.PreProcess(encOutputs, batchSize, g);
            for (int i = 1; i < seqLen; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightTensor> inputs = new List<IWeightTensor>();
                for (int j = 0; j < batchSize; j++)
                {
                    inputs.Add(g.Peek(tgtEmbedding, 0, (int)ix_inputs[j]));
                }
                IWeightTensor inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                IWeightTensor dOutput = decoder.Decode(inputsM, attPreProcessResult, batchSize, g);
                IWeightTensor ffLayer = decoderFFLayer.Process(dOutput, batchSize, g);

                //Softmax for output
                using IWeightTensor probs = g.Softmax(ffLayer, runGradients: false, inPlace: true);
                if (isTraining)
                {
                    //Calculate loss for each word in the batch
                    for (int k = 0; k < batchSize; k++)
                    {
                        float score_k = probs.GetWeightAt(new long[] { k, outputSnts[k][i] });
                        if (i < originalOutputLengths[k])
                        {
                            var lcost = (float)-Math.Log(score_k);
                            if (float.IsNaN(lcost))
                            {
                                throw new ArithmeticException($"Score = '{score_k}' Cost = Nan at index '{i}' word '{outputSnts[k][i]}', Output Sentence = '{String.Join(" ", outputSnts[k])}'");
                            }
                            else
                            {
                                cost += lcost;
                            }
                        }

                        probs.SetWeightAt(score_k - 1, new long[] { k, outputSnts[k][i] });
                        ix_inputs[k] = outputSnts[k][i];
                    }
                    ffLayer.CopyWeightsToGradients(probs);
                }
                else
                {
                    // Output "i"th target word
                    using var targetIdxTensor = g.Argmax(probs, 1);
                    float[] targetIdx = targetIdxTensor.ToWeightArray();
                    for (int j = 0; j < targetIdx.Length; j++)
                    {
                        if (setEndSentId.Contains(j) == false)
                        {
                            outputSnts[j].Add((int)targetIdx[j]);

                            if (targetIdx[j] == eosTokenId)
                            {
                                setEndSentId.Add(j);
                            }
                        }
                    }

                    if (setEndSentId.Count == batchSize)
                    {
                        // All target sentences in current batch are finished, so we exit.
                        break;
                    }

                    ix_inputs = targetIdx;
                }
            }

            return cost;
        }
    }
}
