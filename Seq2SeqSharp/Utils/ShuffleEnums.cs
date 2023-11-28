using System;
using System.Collections.Generic;
using System.Text;

namespace Seq2SeqSharp.Utils
{
    public enum PaddingEnums
    {
        NoPaddingInSrc = 0,
        NoPaddingInTgt,
        NoPadding,
        AllowPadding
    }
}
