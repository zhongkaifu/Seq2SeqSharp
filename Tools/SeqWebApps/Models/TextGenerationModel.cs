// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace SeqWebApps.Models
{
    [Serializable]
    public class TextGenerationModel
    {
        ///<summary>
        /// Gets or sets Output.
        ///</summary>
        public string? Output { get; set; }

        ///<summary>
        /// Gets or sets DateTime.
        ///</summary>
        public string? DateTime { get; set; }
    }
}