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
using Seq2SeqSharp.Tools;
using System.Collections.Generic;
using System.Linq;

namespace Seq2SeqSharp
{
    public class BeamSearchStatus
    {
        public List<int> OutputIds;
        public List<int> AlignmentsToSrc;
        public float Score;

        public BeamSearchStatus()
        {
            OutputIds = new List<int>();
            AlignmentsToSrc = new List<int>();
            Score = 1.0f;
        }
    }

    public class BeamSearch
    {
        public static List<BeamSearchStatus> GetTopNBSS(List<BeamSearchStatus> bssList, int topN)
        {
            FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>> q = new FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>>(topN, new ComparableItemComparer<BeamSearchStatus>(false));

            for (int i = 0; i < bssList.Count; i++)
            {
                q.Enqueue(new ComparableItem<BeamSearchStatus>(bssList[i].Score, bssList[i]));
            }

            return q.Select(x => x.Value).ToList();
        }
    }
}
