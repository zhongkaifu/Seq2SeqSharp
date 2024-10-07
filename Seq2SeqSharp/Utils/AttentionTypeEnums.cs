using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Utils
{
    public enum AttentionTypeEnums
    {
        Classic,
        FlashAttentionV2
    }

    public enum MultiHeadAttentionTypeEnums
    {
        MHA,
        GQA
    }
}
