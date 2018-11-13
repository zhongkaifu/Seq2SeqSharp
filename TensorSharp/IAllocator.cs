using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp
{
    public interface IAllocator
    {
        Storage Allocate(DType elementType, long elementCount);
    }
}
