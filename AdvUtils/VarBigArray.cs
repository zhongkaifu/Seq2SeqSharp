// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace AdvUtils
{
    public sealed class VarBigArray<T> : BigArray<T> where T : IComparable<T>
    {
        long blockSizeInTotal_;
        private object ll = new object();

        public override T this[long offset]
        {
            get
            {
                if (offset >= size_)
                {
                    //resize array size, it need to be synced,
                    //for high performance, we use double check to avoid useless resize call and save memory
                    lock (ll)
                    {
                        if (offset >= size_)
                        {
                            Resize(offset + 1);
                        }
                    }
                }

                long nBlock = offset >> moveBit;
                return arrList[(int)nBlock][offset & (sizePerBlock - 1)];
            }
            set
            {
                if (offset >= size_)
                {
                    //resize array size, it need to be synced,
                    //for high performance, we use double check to avoid useless resize call and save memory
                    lock (ll)
                    {
                        if (offset >= size_)
                        {
                            Resize(offset + 1);
                        }
                    }
                }

                long nBlock = offset >> moveBit;
                arrList[(int)nBlock][offset & (sizePerBlock - 1)] = value;
            }
        }



        private void Resize(long new_size)
        {
            while (blockSizeInTotal_ <= new_size)
            {
                arrList.Add(new T[sizePerBlock]);
                blockSizeInTotal_ += sizePerBlock;
            }

            size_ = new_size;
        }

        //construct variable size big array
        //size is array's default length
        //lowBounding is the lowest bounding of the array
        //when accessing the position which is outer bounding, the big array will be extend automatically.
        public VarBigArray(long size)
        {
            size_ = size;
            arrList = new List<T[]>();

            for (blockSizeInTotal_ = 0; blockSizeInTotal_ < size_;
                blockSizeInTotal_ += sizePerBlock)
            {
                arrList.Add(new T[sizePerBlock]);
            }
        }
    }
}