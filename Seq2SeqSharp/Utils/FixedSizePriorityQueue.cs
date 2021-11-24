using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace AdvUtils
{
    /// <typeparam name="T">T must be comparable.</typeparam>
    public class FixedSizePriorityQueue<T> : IEnumerable<T>
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="capacity">Fix the heap size at this capacity.</param>
        public FixedSizePriorityQueue(int capacity)
        {
            if (capacity < 1)
                throw new Exception("priority queue capacity must be at least one!");
            m_iCount = 0;
            m_iCapacity = capacity;
            m_iBottomSize = m_iTopSize = 0;

            int iBottomCapacity = Math.Max(capacity / 2, 1);
            int iTopCapacity = Math.Max(capacity - iBottomCapacity, 1);

            m_rgtTop = new T[iTopCapacity];
            m_rgtBottom = new T[iBottomCapacity];
        }

        public FixedSizePriorityQueue(int cap, IComparer<T> comp)
            : this(cap)
        {
            comparer = comp;
        }

        #region Basic prio-queue operations.
        /// <summary>
        /// Get the number of elements currently in the queue.
        /// </summary>
        public int Count { get { return m_iCount; } }

        /// <summary>
        /// Get the number of elements that could possibly be held in the queue
        /// </summary>
        public int Capacity { get { return m_iCapacity; } }

        public void Clear()
        {
            m_iCount = 0;
            m_rgtBottom.Initialize();
            m_rgtTop.Initialize();
            m_iTopSize = m_iBottomSize = 0;
        }

        public bool Enqueue(T t)
        {
            // first, are we already at capacity?
            if (m_iCapacity == m_iCount)
            {
                if (m_iCapacity == 1)
                {
                    //We only have a single item in the queue and it's kept in m_rgtTop.
                    if (!Better(t, m_rgtTop[0], true))
                    {
                        return false;
                    }
                    else
                    {
                        m_rgtTop[0] = t;
                        return true;
                    }
                }

                // then, are we better than the bottom?
                if (!Better(t, m_rgtBottom[0], true))
                {
                    // nope, bail.
                    return false;
                }

                // yep, put in place...
                m_rgtBottom[0] = t;
                // first heapfiy the bottom half; get back the
                // index where it ended up.
                int iUpdated = DownHeapify(m_rgtBottom, 0, m_iBottomSize, false);

                // are we not at the boundary?  Then we're done.
                if (!SemiLeaf(iUpdated, m_iBottomSize)) return true;

                // at the boundary: check if we need to update.
                int iTop = CheckBoundaryUpwards(iUpdated);

                // boundary is okay?  bail.
                if (iTop == -1) return true;

                // ...and fix the top heap property.
                UpHeapify(m_rgtTop, iTop, m_iTopSize, true);

                return true;
            }

            // we have space to insert.
            ++m_iCount;
            // need to maintain the invariant that either size(bottom) == size(top),
            // or size(bottom) + 1 == size(top).
            if (m_iBottomSize < m_iTopSize)
            {
                Debug.Assert(m_iBottomSize + 1 == m_iTopSize);
                // bottom is smaller: put it there.
                int iPos = m_iBottomSize++;
                m_rgtBottom[iPos] = t;

                // see if it should really end up in the top heap...
                int iUp = CheckBoundaryUpwards(iPos);

                if (iUp == -1)
                {
                    // no -- fix the bottom yep.
                    UpHeapify(m_rgtBottom, iPos, m_iBottomSize, false);
                }
                else
                {
                    // yes -- fix the top heap.
                    UpHeapify(m_rgtTop, iUp, m_iTopSize, true);
                }
                return true;
            }
            else
            {
                Debug.Assert(m_iBottomSize == m_iTopSize);
                // put it in the top.
                int iPos = m_iTopSize++;
                m_rgtTop[iPos] = t;

                // see if it should really end up in the bottom.
                int iBottom = CheckBoundaryDownwards(iPos);
                if (iBottom == -1)
                {
                    // no -- fix the top heap.
                    UpHeapify(m_rgtTop, iPos, m_iTopSize, true);
                }
                else
                {
                    // yes -- fix the bottotm.
                    UpHeapify(m_rgtBottom, iBottom, m_iBottomSize, false);
                }
                return true;
            }
        }
        #endregion
       
        #region Heap navigators
        int Parent(int i) { return (i - 1) / 2; }
        int Left(int i) { return 2 * i + 1; }
        int Right(int i) { return 2 * i + 2; }
        bool IsLeft(int i) { return i % 2 == 1; }
        bool IsRight(int i) { return i % 2 == 0; }
        bool Leaf(int i, int iSize) { return Left(i) >= iSize; }
        bool SemiLeaf(int i, int iSize) { return Right(i) >= iSize; }

        int BottomNode(int i)
        {
            // first see if we have a direct correspondence.
            if (i < m_iBottomSize) return i;

            // no parallel -- must be that one extra element
            // in the target heap.  instead point at the parent
            // if a left node, or left sibling if a right node.
            Debug.Assert(i <= m_iBottomSize);
            if (i % 2 == 1) return Parent(i);
            return i - 1;
        }

        int TopNode1(int i)
        {
            if (Left(i) >= m_iBottomSize && Left(i) < m_iTopSize) return Left(i);
            // top is always >= bottom in size,
            // so this element is guaranteed to exist.
            return i;
        }

        int TopNode2(int i)
        {
            if (i == m_iBottomSize - 1 &&
                1 == (i % 2) &&
                m_iTopSize > m_iBottomSize)
            {
                return i + 1;
            }
            if (Left(i) >= m_iBottomSize && Left(i) < m_iTopSize) return Left(i);

            return i;
        }
        #endregion

        #region Heap invariant maintenance
        int UpHeapify(T[] rgt, int i, int iSize, bool fTop)
        {
            while (i > 0)
            {
                int iPar = Parent(i);
                if (!Better(rgt[i], rgt[iPar], fTop)) return i;
                Swap(rgt, i, rgt, iPar);
                i = iPar;
            }
            return i;
        }

        int DownHeapify(T[] rgt, int i, int iSize, bool fTop)
        {
            while (true)
            {
                int iLeft = Left(i), iRight = Right(i);
                int iLargest = i;
                if (iLeft < iSize && Better(rgt[iLeft], rgt[iLargest], fTop)) iLargest = iLeft;
                if (iRight < iSize && Better(rgt[iRight], rgt[iLargest], fTop)) iLargest = iRight;
                if (iLargest == i) return i;

                Swap(rgt, i, rgt, iLargest);
                i = iLargest;
            }
        }

        int CheckBoundaryUpwards(int iBottomPos)
        {
            int iTop1 = TopNode1(iBottomPos);
            int iTop2 = TopNode2(iBottomPos);
            int iBetter = -1;
            if (Better(m_rgtBottom[iBottomPos], m_rgtTop[iTop1], true))
            {
                iBetter = iTop1;
            }
            if (Better(m_rgtBottom[iBottomPos], m_rgtTop[iTop2], true) &&
                (iBetter == -1 || Better(m_rgtTop[iTop1], m_rgtTop[iTop2], true)))
            {
                iBetter = iTop2;
            }
            if (iBetter == -1)
            {
                return -1;
            }

            // boundary is not okay? move this guy across...
            Swap(m_rgtTop, iBetter, m_rgtBottom, iBottomPos);

            return iBetter;
        }

        int CheckBoundaryDownwards(int iTopPos)
        {
            // compare to the bottom guy in the corresponding posn.
            int iBottomPos = BottomNode(iTopPos);
            if (iBottomPos == -1)
            {
                return -1;
            }
            if (iBottomPos >= m_iBottomSize ||
                !Better(m_rgtBottom[iBottomPos], m_rgtTop[iTopPos], true))
            {
                return -1;
            }

            Swap(m_rgtTop, iTopPos, m_rgtBottom, iBottomPos);

            return iBottomPos;
        }

        void Swap(T[] rgt1, int i1, T[] rgt2, int i2)
        {
            T tTemp = rgt1[i1];
            rgt1[i1] = rgt2[i2];
            rgt2[i2] = tTemp;
        }

        protected bool Better(T t1, T t2, bool fTop)
        {
            int i = (comparer == null) ? ((IComparable<T>)t1).CompareTo(t2) : comparer.Compare(t1, t2);
            //int i = comparer.Compare(t1, t2);

            return (!fTop) ? i < 0 : i > 0;

        }
        #endregion

        #region Data structures
        /// <summary>
        /// The downward facing heap.
        /// </summary>
        protected T[] m_rgtBottom;
        /// <summary>
        /// Upward facing heap.
        /// </summary>
        protected T[] m_rgtTop;
        /// <summary>
        /// Total number of elements in the heap.
        /// </summary>
        protected int m_iCount;
        /// <summary>
        /// Capacity of the heap.
        /// </summary>
        protected int m_iCapacity;
        /// <summary>
        /// Number of nodes in the bottom heap.
        /// </summary>
        protected int m_iBottomSize;
        /// <summary>
        /// Number of nodes in the top heap.
        /// </summary>
        protected int m_iTopSize;
        #endregion

        protected IComparer<T> comparer = null;

        #region IEnumerable<T> Members

        public IEnumerator<T> GetEnumerator()
        {
            for (int i = 0; i < m_iTopSize; ++i)
                yield return m_rgtTop[i];
            for (int i = m_iBottomSize - 1; i >= 0; --i)
                yield return m_rgtBottom[i];
        }

        #endregion

        #region IEnumerable Members

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        #endregion

    }
}

