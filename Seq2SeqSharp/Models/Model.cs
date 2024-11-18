// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;

using AdvUtils;
using ManagedCuda.BasicTypes;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Enums;
using TensorSharp;
using System.Runtime.InteropServices;
using System.Data;

namespace Seq2SeqSharp.Models
{
    [StructLayout(LayoutKind.Explicit)]
    public class Name2WeightsHalf
    {
        [FieldOffset(0)] public Dictionary<string, ushort[]> usDict;
        [FieldOffset(0)] public Dictionary<string, half[]> halfDict;

        public Name2WeightsHalf()
        {
            usDict = new Dictionary<string, ushort[]>();
        }

        public void Clear()
        {
            if (usDict != null)
            {
                usDict.Clear();
            }
            else
            {
                usDict = new Dictionary<string, ushort[]>();
            }
        }
    }

    [Serializable]
    public abstract class Model : IModel
    {
        public int DecoderEmbeddingDim { get; set; }
        public int EncoderEmbeddingDim { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int EncoderLayerDepth { get; set; }

        public ActivateFuncEnums ActivateFunc { get; set; }

        public int ExpertNum { get; set; }
        public int ExpertsPerTokenFactor { get; set; }
        public DecoderTypeEnums DecoderType { get; set; }
        public EncoderTypeEnums EncoderType { get; set; }
        public int HiddenDim { get; set; }

        public int IntermediateDim { get; set; }
        public bool EnableSegmentEmbeddings { get; set; }
        public int MultiHeadNum { get; set; }
        public Vocab SrcVocab { get; set; }
        public Vocab TgtVocab { get; set; }
		public List<Vocab> ClsVocabs { get; set; }
        public bool EnableCoverageModel { get; set; }
        public bool SharedEmbeddings { get; set; }

        public string SimilarityType { get; set; }

        public bool EnableTagEmbeddings { get; set; }

        public int MaxSegmentNum { get; set; }

        public bool PointerGenerator { get; set; }


        public Dictionary<string, float[]> Name2Weights { get; set; }

        public Name2WeightsHalf Name2WeightsHalf { get; set; }

        public VQTypeEnums VQType { get; set; }
        public Dictionary<string, byte[]> Name2WeightsVQ { get; set; }       
        public Dictionary<string, double[]> Name2CodeBook { get; set; }

        public PositionEmbeddingEnums PEType { get; set; }
        public NormEnums NormType { get; set; }

        public MultiHeadAttentionTypeEnums MultiHeadAttentionType { get; set; }
        public int KVGroupNum { get; set; }

        public Model() { }
        public Model(Options opts,Vocab srcVocab, Vocab tgtVocab)
        {
            HiddenDim = opts.HiddenSize;
            IntermediateDim = opts.IntermediateSize;
            EncoderLayerDepth = opts.EncoderLayerDepth;;
            EncoderType = opts.EncoderType;
            MultiHeadNum = opts.MultiHeadNum;
            SrcVocab = srcVocab;
            TgtVocab = tgtVocab;
            EncoderEmbeddingDim = opts.SrcEmbeddingDim;
            EnableSegmentEmbeddings = opts.EnableSegmentEmbeddings;
            EnableTagEmbeddings = opts.EnableTagEmbeddings;
            MaxSegmentNum = opts.MaxSegmentNum;
            ExpertNum = opts.ExpertNum;
            ExpertsPerTokenFactor = opts.ExpertsPerTokenFactor;
            ActivateFunc = opts.ActivateFunc;
            VQType = opts.VQType;
            PEType = opts.PEType;
            NormType = opts.NormType;
            MultiHeadAttentionType = opts.MultiHeadAttentionType;
            KVGroupNum = opts.KVGroupNum;

            Name2Weights = new Dictionary<string, float[]>();
            Name2WeightsHalf = new Name2WeightsHalf();
            Name2WeightsVQ = new Dictionary<string, byte[]>();
            Name2CodeBook = new Dictionary<string, double[]>();
        }

        public Model(Model_4_ProtoBufSerializer m)
        {
            HiddenDim = m.HiddenDim;
            IntermediateDim = m.IntermediateDim;
            EncoderLayerDepth = m.EncoderLayerDepth; ;
            EncoderType = m.EncoderType;
            MultiHeadNum = m.MultiHeadNum;
            SrcVocab = m.SrcVocab?.ToVocab();
            TgtVocab= m.TgtVocab?.ToVocab();
            EncoderEmbeddingDim = m.EncoderEmbeddingDim;
            EnableSegmentEmbeddings = m.EnableSegmentEmbeddings;
            EnableTagEmbeddings = m.EnableTagEmbeddings;
            MaxSegmentNum = m.MaxSegmentNum;
            ExpertNum = m.ExpertNum;
            ExpertsPerTokenFactor = m.ExpertsPerTokenFactor;
            SimilarityType = m.SimilarityType;
            ActivateFunc = m.ActivateFunc;
            VQType = m.VQType;

            Name2Weights = m.Name2Weights;
            Name2WeightsHalf = new Name2WeightsHalf();
            Name2WeightsHalf.usDict = m.Name2WeightsHalf;
            Name2WeightsVQ = m.Name2WeightsVQ;
            Name2CodeBook = m.Name2CodeBook;
            PEType = m.PEType;
            NormType = m.NormType;

            MultiHeadAttentionType = m.MultiHeadAttentionType;
            KVGroupNum = m.KVGroupNum;


            if (Name2Weights == null)
            {
                Name2Weights = new Dictionary<string, float[]>();
            }

            if (Name2WeightsHalf.usDict == null)
            {
                Name2WeightsHalf.usDict = new Dictionary<string, ushort[]>();
            }

            if (Name2WeightsVQ == null)
            {
                Name2WeightsVQ = new Dictionary<string, byte[]>();
            }

            if (Name2CodeBook == null)
            {
                Name2CodeBook = new Dictionary<string, double[]>();
            }
        }

        public void AddWeights(string name, float[] weights)
        {
            Logger.WriteLine(Logger.Level.debug, $"Adding weights '{name}' to the model.");

            if (VQType == VQTypeEnums.FLOAT16)
            {             
                var weightsHalf = new ushort[weights.Length];
                for (int i = 0; i < weights.Length; i++)
                {
                    weightsHalf[i] = (new half(weights[i])).x;
                }
                Name2WeightsHalf.usDict.Add(name, weightsHalf);
            }
            else if (VQType == VQTypeEnums.INT8)
            {
                int vqSize = 256;
                VectorQuantization vq = new VectorQuantization();
                foreach (var v in weights)
                {
                    vq.Add(v);
                }

                for (int i = weights.Length; i < vqSize; i++)
                {
                    vq.Add(0);
                }

                (double min, double max, double distortion) = vq.BuildCodebook(vqSize);

                Name2CodeBook.Add(name, vq.CodeBook);

                byte[] bweights = new byte[weights.Length];
                for (int i = 0; i < weights.Length; i++)
                {
                    bweights[i] = (byte)vq.ComputeVQ(weights[i]);
                }

                Name2WeightsVQ.Add(name, bweights);
            }
            else if (VQType == VQTypeEnums.INT4)
            {
                int vqSize = 16;
                VectorQuantization vq = new VectorQuantization();
                foreach (var v in weights)
                {
                    vq.Add(v);
                }

                for (int i = weights.Length; i < vqSize; i++)
                {
                    vq.Add(0);
                }

                (double min, double max, double distortion) = vq.BuildCodebook(vqSize);
             //   if (Math.Abs(max - min) <= 2.0)
             //   {
                    Name2CodeBook.Add(name, vq.CodeBook);

                    byte[] bweights = new byte[weights.Length / 2];
                    for (int i = 0; i < weights.Length; i += 2)
                    {
                        int lowWeight = vq.ComputeVQ(weights[i]);
                        int highWeight = vq.ComputeVQ(weights[i + 1]);

                        bweights[i / 2] = (byte)(highWeight * 16 + lowWeight);
                    }

                    Name2WeightsVQ.Add(name, bweights);
             //   }
             //   else
             //   {
             //       Logger.WriteLine($"Distortion({distortion}) is too large, so we keep the original values.");
             //       Name2Weights.Add(name, weights);
             //   }
            }
            else
            {
                Name2Weights.Add(name, weights);
            }
        }

        public float[] GetWeights(string name)
        {
            float[] weight = null;

            if (Name2Weights.ContainsKey(name))
            {
                weight = Name2Weights[name];
            }
            else if (Name2WeightsHalf.halfDict.ContainsKey(name))
            {
                throw new InvalidCastException($"The model is saved as Float16 type, so please enable AMP for model loading.");
            }
            else if (VQType == VQTypeEnums.INT8)
            {
                if (Name2WeightsVQ.ContainsKey(name) == false)
                {
                    Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
                    return null;
                }

                var codeBook = Name2CodeBook[name];

                weight = new float[Name2WeightsVQ[name].Length];
                for (int i = 0; i < Name2WeightsVQ[name].Length; i++)
                {
                    weight[i] = (float)codeBook[Name2WeightsVQ[name][i]];
                }
            }
            else if (VQType == VQTypeEnums.INT4)
            {
                if (Name2WeightsVQ.ContainsKey(name) == false)
                {
                    Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
                    return null;
                }

                var codeBook = Name2CodeBook[name];

                weight = new float[Name2WeightsVQ[name].Length * 2];
                for (int i = 0; i < Name2WeightsVQ[name].Length; i++)
                {
                    double highWeight = codeBook[Name2WeightsVQ[name][i] / 16];
                    double lowWeight = codeBook[Name2WeightsVQ[name][i] & 0x0F];

                    weight[i * 2] = (float)lowWeight;
                    weight[i * 2 + 1] = (float)highWeight;
                }

            }
            else
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
            }

            return weight;
        }

        public half[] GetWeightsHalfType(string name)
        {
            half[] weights = null;
            if (Name2Weights.ContainsKey(name))
            {
                var values = Name2Weights[name];
                weights = new half[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    weights[i] = new half(values[i]);
                }
            }
            else if (Name2WeightsHalf.halfDict.ContainsKey(name))
            {
                weights = Name2WeightsHalf.halfDict[name];

                //var values = Name2WeightsHalf[name];
                //weights = new half[values.Length];
                //for (int i = 0; i < values.Length; i++)
                //{
                //    weights[i] = new half(values[i]);
                //}
            }
            else if (VQType == VQTypeEnums.INT8)
            {
                if (Name2WeightsVQ.ContainsKey(name) == false)
                {
                    Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
                    return null;
                }

                var codeBook = Name2CodeBook[name];

                weights = new half[Name2WeightsVQ[name].Length];
                for (int i = 0; i < Name2WeightsVQ[name].Length; i++)
                {
                    weights[i] = new half(codeBook[Name2WeightsVQ[name][i]]);
                }
            }
            else if (VQType == VQTypeEnums.INT4)
            {
                if (Name2WeightsVQ.ContainsKey(name) == false)
                {
                    Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
                    return null;
                }

                var codeBook = Name2CodeBook[name];

                weights = new half[Name2WeightsVQ[name].Length * 2];
                for (int i = 0; i < Name2WeightsVQ[name].Length; i++)
                {
                    double highWeight = codeBook[Name2WeightsVQ[name][i] / 16];
                    double lowWeight = codeBook[Name2WeightsVQ[name][i] & 0x0F];

                    weights[i * 2] = new half(lowWeight);
                    weights[i * 2 + 1] = new half(highWeight);
                }
            }
            else
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
            }

            return weights;
        }

        public void DeleteWeights(string name)
        {
            if (Name2WeightsVQ != null && Name2WeightsVQ.ContainsKey(name))
            {
                Name2WeightsVQ.Remove(name);
            }

            if (Name2CodeBook != null && Name2CodeBook.ContainsKey(name))
            {
                Name2CodeBook.Remove(name);
            }

            if (Name2Weights != null && Name2Weights.ContainsKey(name))
            {
                Name2Weights.Remove(name);
            }

            if (Name2WeightsHalf != null && Name2WeightsHalf.halfDict.ContainsKey(name))
            {
                Name2WeightsHalf.halfDict.Remove(name);
            }
        }

        public void ClearWeights()
        {
            Name2WeightsVQ.Clear();
            Name2CodeBook.Clear();
            Name2Weights.Clear();
            Name2WeightsHalf.Clear();
        }

        public void ShowModelInfo()
        {
#if DEBUG           
            Logger.WriteLine(Logger.Level.debug, $"Encoder embedding dim: '{EncoderEmbeddingDim}'");
            Logger.WriteLine(Logger.Level.debug, $"Decoder embedding dim: '{DecoderEmbeddingDim}'");
            Logger.WriteLine(Logger.Level.debug, $"Encoder layer depth: '{EncoderLayerDepth}'");
            Logger.WriteLine(Logger.Level.debug, $"Decoder layer depth: '{DecoderLayerDepth}'");
            Logger.WriteLine(Logger.Level.debug, $"Encoder type: '{EncoderType}'");
            Logger.WriteLine(Logger.Level.debug, $"Decoder type: '{DecoderType}'");
            Logger.WriteLine(Logger.Level.debug, $"Hidden layer dim: '{HiddenDim}'");
            Logger.WriteLine(Logger.Level.debug, $"Intermediate dim: '{IntermediateDim}");
            Logger.WriteLine(Logger.Level.debug, $"Enable segment embeddings: '{EnableSegmentEmbeddings}'");
            Logger.WriteLine(Logger.Level.debug, $"Enable shared embeddings: '{SharedEmbeddings}'");
            Logger.WriteLine(Logger.Level.debug, $"Enable tag embeddings: '{EnableTagEmbeddings}'");
            Logger.WriteLine(Logger.Level.debug, $"Multi-head size: '{MultiHeadNum}'");
            Logger.WriteLine(Logger.Level.debug, $"Pointer Generator: '{PointerGenerator}'");
            Logger.WriteLine(Logger.Level.debug, $"Expert Size: '{ExpertNum}");
            Logger.WriteLine(Logger.Level.debug, $"Experts per token factor: '{ExpertsPerTokenFactor}'");
            Logger.WriteLine(Logger.Level.debug, $"Codebook size for model vector quantization: '{VQType}'");
            Logger.WriteLine(Logger.Level.debug, $"Positional Embedding Type: '{PEType}'");


            if (!SimilarityType.IsNullOrEmpty())
            {
                Logger.WriteLine(Logger.Level.debug, $"Similarity Type: '{SimilarityType}'");
            }

            if (SrcVocab != null)
            {
                Logger.WriteLine(Logger.Level.debug, $"Source vocabulary size: '{SrcVocab.Count}'");
            }

            if (TgtVocab != null)
            {
                Logger.WriteLine(Logger.Level.debug, $"Target vocabulary size: '{TgtVocab.Count}'");
            }
			
            if (ClsVocabs != null)
            {
                Logger.WriteLine(Logger.Level.debug, $"Converting old model cls vocab to tgt vocab.");
            }
#endif
        }
    }
}
