// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace Seq2SeqSharp.Enums
{
    public enum ModeEnums
    {
        Train,
        Valid,
        Test,
        Alignment,
        DumpVocab,
        UpdateVocab,
        VQModel,
        Help
    }

    public enum EncoderTypeEnums
    {
        None = -1,
        BiLSTM = 0,
        Transformer = 1,
    }

    public enum DecoderTypeEnums
    {
        None = -1,
        AttentionLSTM = 0,
        Transformer = 1,
        GPTDecoder = 2
    }

    public enum VQTypeEnums
    {
        None = 0,
        FLOAT16 = 65536,
        INT8 = 256,
        INT4 = 16
    }

    public enum NormEnums
    {
        LayerNorm = 0,
        RMSNorm = 1
    }
}
