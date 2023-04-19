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

namespace Seq2SeqSharp.Tools
{

    public class ComparableItem<T>
    {
        public float Score { get; }
        public T Value { get; }

        public ComparableItem(float score, T value)
        {
            Score = score;
            Value = value;
        }
    }

    public class ComparableItemComparer<T> : IComparer<ComparableItem<T>>
    {
        public ComparableItemComparer(bool fAscending)
        {
            m_fAscending = fAscending;
        }

        public int Compare(ComparableItem<T> x, ComparableItem<T> y)
        {
            int iSign = Math.Sign(x.Score - y.Score);
            if (!m_fAscending)
            {
                iSign = -iSign;
            }

            return iSign;
        }

        protected bool m_fAscending;
    }
}
