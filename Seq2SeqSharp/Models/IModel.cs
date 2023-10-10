// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using ManagedCuda.BasicTypes;
using Seq2SeqSharp.Utils;
using Seq2SeqSharp.Enums;
using System.Collections.Generic;

namespace Seq2SeqSharp
{
    public interface IModel
    {
        public int DecoderEmbeddingDim { get; set; }
        public int EncoderEmbeddingDim { get; set; }
        public int DecoderLayerDepth { get; set; }
        public int EncoderLayerDepth { get; set; }
        public VQTypeEnums VQType { get; set; }
        public PositionEmbeddingEnums PEType { get; set; }
        public NormEnums NormType { get; set; }
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
        public bool EnableCoverageModel { get; set; }
        public bool SharedEmbeddings { get; set; }

        public string SimilarityType { get; set; }

        public bool EnableTagEmbeddings { get; set; }

        public int MaxSegmentNum { get; set; }

        public void AddWeights(string name, float[] weights);

        public float[] GetWeights(string name);
        half[] GetWeightsHalfType(string name);

        public void DeleteWeights(string name);

        public bool PointerGenerator { get; set; }

        public void ClearWeights();

        public void ShowModelInfo();
    }
}
