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
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Utils;

namespace Seq2SeqSharp.Models
{
    [Serializable]
    public abstract class Model : IModel
    {
        public int DecoderEmbeddingDim { get; set; }
        public int EncoderEmbeddingDim { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int EncoderLayerDepth { get; set; }

        public int ExpertNum { get; set; }
        public int ExpertsPerTokenFactor { get; set; }
        public DecoderTypeEnums DecoderType { get; set; }
        public EncoderTypeEnums EncoderType { get; set; }
        public int HiddenDim { get; set; }
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

        public Vocab ClsVocab
        {
            get
            {
                if (ClsVocabs == null)
                {
                    ClsVocabs = new List<Vocab>
                    {
                        new Vocab()
                    };
                }

                return ClsVocabs[0];
            }

            set
            {
                if (ClsVocabs == null)
                {
                    ClsVocabs = new List<Vocab>
                    {
                        new Vocab()
                    };
                }

                ClsVocabs[0] = value;
            }
        }


        public Dictionary<string, float[]> Name2Weights { get; set; }

        public Model() { }
        public Model(Options opts,Vocab srcVocab)
        {
            HiddenDim = opts.HiddenSize;
            EncoderLayerDepth = opts.EncoderLayerDepth;;
            EncoderType = opts.EncoderType;
            MultiHeadNum = opts.MultiHeadNum;
            SrcVocab = srcVocab;
            EncoderEmbeddingDim = opts.SrcEmbeddingDim;
            EnableSegmentEmbeddings = opts.EnableSegmentEmbeddings;
            EnableTagEmbeddings = opts.EnableTagEmbeddings;
            MaxSegmentNum = opts.MaxSegmentNum;
            ExpertNum = opts.ExpertNum;
            ExpertsPerTokenFactor = opts.ExpertsPerTokenFactor;

            Name2Weights = new Dictionary<string, float[]>();
        }

        public Model(Model_4_ProtoBufSerializer m)
        {
            HiddenDim = m.HiddenDim;
            EncoderLayerDepth = m.EncoderLayerDepth; ;
            EncoderType = m.EncoderType;
            MultiHeadNum = m.MultiHeadNum;
            SrcVocab = m.SrcVocab?.ToVocab();
            EncoderEmbeddingDim = m.EncoderEmbeddingDim;
            EnableSegmentEmbeddings = m.EnableSegmentEmbeddings;
            EnableTagEmbeddings = m.EnableTagEmbeddings;
            MaxSegmentNum = m.MaxSegmentNum;
            ExpertNum = m.ExpertNum;
            ExpertsPerTokenFactor = m.ExpertsPerTokenFactor;
            SimilarityType = m.SimilarityType;

            Name2Weights = new Dictionary<string, float[]>();

        }

        public void AddWeights(string name, float[] weights)
        {
            Name2Weights.Add(name, weights);
        }

        public float[] GetWeights(string name)
        {
            if (Name2Weights.ContainsKey(name) == false)
            {
                Logger.WriteLine(Logger.Level.warn, ConsoleColor.Yellow, $"Weight '{name}' doesn't exist in the model.");
                return null;
            }

            return Name2Weights[name];
        }

        public void ClearWeights()
        {
            Name2Weights.Clear();
        }

        public void ShowModelInfo()
        {
            Logger.WriteLine($"Encoder embedding dim: '{EncoderEmbeddingDim}'");
            Logger.WriteLine($"Decoder embedding dim: '{DecoderEmbeddingDim}'");
            Logger.WriteLine($"Encoder layer depth: '{EncoderLayerDepth}'");
            Logger.WriteLine($"Decoder layer depth: '{DecoderLayerDepth}'");
            Logger.WriteLine($"Encoder type: '{EncoderType}'");
            Logger.WriteLine($"Decoder type: '{DecoderType}'");
            Logger.WriteLine($"Hidden layer dim: '{HiddenDim}'");
            Logger.WriteLine($"Enable segment embeddings: '{EnableSegmentEmbeddings}'");
            Logger.WriteLine($"Enable shared embeddings: '{SharedEmbeddings}'");
            Logger.WriteLine($"Enable tag embeddings: '{EnableTagEmbeddings}'");
            Logger.WriteLine($"Multi-head size: '{MultiHeadNum}'");
            Logger.WriteLine($"Pointer Generator: '{PointerGenerator}'");
            Logger.WriteLine($"Expert Size: '{ExpertNum}");
            Logger.WriteLine($"Experts per token factor: '{ExpertsPerTokenFactor}'");


            if (!SimilarityType.IsNullOrEmpty())
            {
                Logger.WriteLine($"Similarity Type: '{SimilarityType}'");
            }

            if (SrcVocab != null)
            {
                Logger.WriteLine($"Source vocabulary size: '{SrcVocab.Count}'");
            }

            if (TgtVocab != null)
            {
                Logger.WriteLine($"Target vocabulary size: '{TgtVocab.Count}'");
            }

            if (ClsVocabs != null)
            {
                Logger.WriteLine($"The number of CLS vocabularies: '{ClsVocabs.Count}' ");
                for (int i = 0; i < ClsVocabs.Count; i++)
                {
                    Logger.WriteLine($"CLS vocabulary {i} size: {ClsVocabs[i].Count}");
                }
            }
        }
    }
}
