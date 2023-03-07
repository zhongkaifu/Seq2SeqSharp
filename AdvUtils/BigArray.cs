// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

#if NO_SUPPORT_PARALLEL_LIB
#else
using System.Threading.Tasks;
#endif

namespace AdvUtils
{
    abstract public class BigArray<T> : IList<T> where T : IComparable<T>
    {
        public const long sizePerBlock = 1024 * 1024 * 64; //(<<26bits)
        public const int moveBit = 26;
        public long size_;
        public List<T[]> arrList;


        public BigArray()
        {

        }

        public int IndexOf(T item)
        {
            throw new NotImplementedException();
        }

        public void Insert(int index, T item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public T this[int i]
        {
            get
            {
                return this[(long)i];
            }
            set
            {
                this[(long)i] = value;
            }
        }

        public abstract T this[long i]
        {
            get;
            set;
        }

        public void Add(T item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            foreach (T[] item in arrList)
            {
                Array.Clear(item, 0, item.Length);
            }
        }

        public bool Contains(T item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public int Count
        {
            get
            {
                return (int)LongLength;
            }
        }

        public long LongLength
        {
            get
            {
                return size_;
            }
        }

        public bool IsReadOnly
        {
            get { throw new NotImplementedException(); }
        }

        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            throw new NotImplementedException();
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            throw new NotImplementedException();
        }


        void swap(long pos1, long pos2)
        {
            int nBlock1 = (int)(pos1 >> moveBit);
            int offset1 = (int)(pos1 & (sizePerBlock - 1));

            int nBlock2 = (int)(pos2 >> moveBit);
            int offset2 = (int)(pos2 & (sizePerBlock - 1));

            T tmp = arrList[nBlock1][offset1];
            arrList[nBlock1][offset1] = arrList[nBlock2][offset2];
            arrList[nBlock2][offset2] = tmp;
        }


        private long med3(long a, long b, long c)
        {
            return this[a].CompareTo(this[b]) < 0 ? (this[b].CompareTo(this[c]) < 0 ? b : this[a].CompareTo(this[c]) < 0 ? c : a) : this[b].CompareTo(this[c]) > 0 ? b : this[a].CompareTo(this[c]) > 0 ? c : a;
        }

        private void vecswap(long a, long b, long n)
        {

            for (long i = 0;i < n;i++)

            {
                int nBlock1 = (int)((a + i) >> moveBit);
                int offset1 = (int)((a + i) & (sizePerBlock - 1));

                int nBlock2 = (int)((b + i) >> moveBit);
                int offset2 = (int)((b + i) & (sizePerBlock - 1));

                T tmp = arrList[nBlock1][offset1];
                arrList[nBlock1][offset1] = arrList[nBlock2][offset2];
                arrList[nBlock2][offset2] = tmp;
            }

        }

        const int INSERT_SORT_THRESHOLD = 7;
        public void QuickSort(long left, long right)
        {
            if (left >= right)
            {
                return;
            }

            //use insert sort to handle small data
            long len = right - left + 1;
            if (len < INSERT_SORT_THRESHOLD)
            {
                for (long i = left; i <= right; i++)
                {
                    T t = this[i];
                    long j = i;
                    for (; j > left && this[j - 1].CompareTo(t) > 0; j--)
                    {
                        this[j] = this[j - 1];
                    }
                    this[j] = t;
                }
                return;
            }

            //Choose the pivot value
            long mid = left + (len >> 1);
            if (len > INSERT_SORT_THRESHOLD)
            {
                //Split the list into three parts, and find middle value in each part,
                //and finally, use middle value in above three middle values as pivot.
                long leftMid = left;
                long rightMid = right;
                if (len > 40)
                {
                    long size = len / 8;
                    leftMid = med3(leftMid, leftMid + size, leftMid + 2 * size);
                    mid = med3(mid - size, mid, mid + size);
                    rightMid = med3(right - 2 * size, right - size, right);
                }
                mid = med3(leftMid, mid, rightMid);
            }

            T v = this[mid];

            //Scan the list from two directions
            long pivotLeftSide = left, leftScanIndex = pivotLeftSide;
            long rightScanIndex = right, pivotRightSide = rightScanIndex;


            int leftScanIndexBlock = (int)(leftScanIndex >> moveBit);
            int leftScanIndexOffset = (int)(leftScanIndex & (sizePerBlock - 1));
            T[] arrayLeft = arrList[leftScanIndexBlock];

            int rightScanIndexBlock = (int)(rightScanIndex >> moveBit);
            int rightScanIndexOffset = (int)(rightScanIndex & (sizePerBlock - 1));
            T[] arrayRight = arrList[rightScanIndexBlock];
            while (true)
            {
                //Try to find item which is bigger than pivot
                while (leftScanIndex <= rightScanIndex)
                {
                    int cmpRst = arrayLeft[leftScanIndexOffset].CompareTo(v);
                    if (cmpRst > 0)
                    {
                        //Found one.
                        break;
                    }
                    else if (cmpRst == 0)
                    {
                        //If the item is equal to pivot, exchange it with the item in left-side.
                        swap(pivotLeftSide++, leftScanIndex);
                    }
                    leftScanIndex++;

                    leftScanIndexOffset++;
                    if (leftScanIndexOffset == sizePerBlock)
                    {
                        leftScanIndexOffset = 0;
                        leftScanIndexBlock++;
                        if (leftScanIndexBlock == arrList.Count)
                        {
                            break;
                        }
                        arrayLeft = arrList[leftScanIndexBlock];
                    }
                }

                //Try to find item which is smaller than pivot
                while (rightScanIndex >= leftScanIndex)
                {
                    int cmpRst = arrayRight[rightScanIndexOffset].CompareTo(v);
                    if (cmpRst < 0)
                    {
                        //Found one.
                        break;
                    }
                    else if (cmpRst == 0)
                    {
                        //If the item is equal to pivot, exchange it with the item in left-side.
                        swap(rightScanIndex, pivotRightSide--);
                    }
                    rightScanIndex--;

                    rightScanIndexOffset--;
                    if (rightScanIndexOffset < 0)
                    {
                        rightScanIndexOffset = (int)(sizePerBlock - 1);
                        rightScanIndexBlock--;
                        if (rightScanIndexBlock < 0)
                        {
                            break;
                        }
                        arrayRight = arrList[rightScanIndexBlock];
                    }
                }

                if (leftScanIndex > rightScanIndex)
                {
                    //Scan finished
                    break;
                }

                //Exchange two found items between pivot
                T temp = arrayLeft[(int)leftScanIndexOffset];
                arrayLeft[(int)leftScanIndexOffset] = arrayRight[(int)rightScanIndexOffset];
                arrayRight[(int)rightScanIndexOffset] = temp;

                leftScanIndex++;
                rightScanIndex--;

                leftScanIndexOffset++;
                if (leftScanIndexOffset == sizePerBlock)
                {
                    leftScanIndexOffset = 0;
                    leftScanIndexBlock++;
                    if (leftScanIndexBlock == arrList.Count)
                    {
                        break;
                    }
                    arrayLeft = arrList[leftScanIndexBlock];
                }

                rightScanIndexOffset--;
                if (rightScanIndexOffset < 0)
                {
                    rightScanIndexOffset = (int)(sizePerBlock - 1);
                    rightScanIndexBlock--;
                    if (rightScanIndexBlock < 0)
                    {
                        break;
                    }
                    arrayRight = arrList[rightScanIndexBlock];
                }
            }

            //Continue to sort two sub-sections
            long splitIndexLeft = leftScanIndex - pivotLeftSide;
            long splitIndexRight = pivotRightSide - rightScanIndex;
            if (splitIndexLeft > 1 && splitIndexRight > 1)
            {

                    {
                        //exchange items with same value into middle of the list
                        long size = Math.Min(pivotLeftSide - left, leftScanIndex - pivotLeftSide);
                        vecswap(left, leftScanIndex - size, size);
                        QuickSort(left, splitIndexLeft + left - 1);
                    }

                    {
                        //exchange items with same value into middle of the list
                        long size = Math.Min(pivotRightSide - rightScanIndex, right - pivotRightSide);
                        vecswap(leftScanIndex, right - size + 1, size);
                        QuickSort(right - splitIndexRight + 1, right);
                    }

            }
            else
            {
                //exchange items with same value into middle of the list
                long size = Math.Min(pivotLeftSide - left, leftScanIndex - pivotLeftSide);
                vecswap(left, leftScanIndex - size, size);

                size = Math.Min(pivotRightSide - rightScanIndex, right - pivotRightSide);
                vecswap(leftScanIndex, right - size + 1, size);

                if (splitIndexLeft > 1)
                {
                    QuickSort(left, splitIndexLeft + left - 1);
                }

                if (splitIndexRight > 1)
                {
                    QuickSort(right - splitIndexRight + 1, right);
                }
            }
        }



        public void Sort(long startIndex, long size, int threadnum = -1)
        {
            QuickSort(startIndex, startIndex + size - 1);
        }

        public void Sort(int threadnum = -1)
        {
            QuickSort(0, Count - 1);
        }


        public T BinarySearch(long low, long high, T goal)
        {
            long mid = 0;

            while (low <= high)
            {
                mid = (high + low) / 2;
                if (this[mid].CompareTo(goal) == 0)
                {
                    return this[mid];
                }
                else if (this[mid].CompareTo(goal) > 0)
                {
                    high = mid - 1;
                }
                else
                {
                    low = mid + 1;
                }
            }
            return default(T);
        }
    }
}