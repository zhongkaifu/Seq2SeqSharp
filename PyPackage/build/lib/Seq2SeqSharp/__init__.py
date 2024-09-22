from pythonnet import load
load("coreclr")
import clr
import os
import sys
from System import Type

import ctypes
import numpy as np
import System
from System import Array, Int32
from System.Runtime.InteropServices import GCHandle, GCHandleType


dir = os.path.dirname(sys.modules["Seq2SeqSharp"].__file__)
path = os.path.join(dir, "Seq2SeqSharp.dll")
clr.AddReference(path)

from Seq2SeqSharp.LearningRate import DecayLearningRate
from Seq2SeqSharp.Applications import DecodingOptions, Seq2SeqOptions, GPT, Seq2Seq
from Seq2SeqSharp.Corpus import Seq2SeqCorpusBatch, Seq2SeqCorpus, SeqCorpus
from Seq2SeqSharp.Utils import TensorUtils, ProcessorTypeEnums, Misc, PaddingEnums, Vocab, DecodingStrategyEnums
from Seq2SeqSharp.Enums import ModeEnums, DecoderTypeEnums
from Seq2SeqSharp.Metrics import BleuMetric
from Seq2SeqSharp.Tools import TooLongSequence, WeightTensor, WeightTensorFactory, ComputeGraphTensor
from Seq2SeqSharp.Optimizer import AdamOptimizer


_MAP_NP_NET = {
    np.dtype(np.float32): System.Single,
    np.dtype(np.float64): System.Double,
    np.dtype(np.int8)   : System.SByte,
    np.dtype(np.int16)  : System.Int16,
    np.dtype(np.int32)  : System.Int32,
    np.dtype(np.int64)  : System.Int64,
    np.dtype(np.uint8)  : System.Byte,
    np.dtype(np.uint16) : System.UInt16,
    np.dtype(np.uint32) : System.UInt32,
    np.dtype(np.uint64) : System.UInt64
}
_MAP_NET_NP = {
    'Single' : np.dtype(np.float32),
    'Double' : np.dtype(np.float64),
    'SByte'  : np.dtype(np.int8),
    'Int16'  : np.dtype(np.int16), 
    'Int32'  : np.dtype(np.int32),
    'Int64'  : np.dtype(np.int64),
    'Byte'   : np.dtype(np.uint8),
    'UInt16' : np.dtype(np.uint16),
    'UInt32' : np.dtype(np.uint32),
    'UInt64' : np.dtype(np.uint64)
}

def asNumpyArray(netArray: System.Array):
    """
    Converts a .NET array to a NumPy array. See `_MAP_NET_NP` for 
    the mapping of CLR types to Numpy ``dtype``.

    Parameters
    ----------
    netArray: System.Array
        The array to be converted

    Returns
    -------
    numpy.ndarray 
    """
    dims = np.empty(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    netType = netArray.GetType().GetElementType().Name

    try:
        npArray = np.empty(dims, order='C', dtype=_MAP_NET_NP[netType])
    except KeyError:
        raise NotImplementedError(f'asNumpyArray does support System type {netType}')

    try: # Memmove 
        sourceHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = sourceHandle.AddrOfPinnedObject().ToInt64()
        destPtr = npArray.__array_interface__['data'][0]
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if sourceHandle.IsAllocated: 
            sourceHandle.Free()
    return npArray


def asNetArray(npArray):
    """
    Converts a NumPy array to a .NET array. See `_MAP_NP_NET` for 
    the mapping of CLR types to Numpy ``dtype``.

    Parameters
    ----------
    npArray: numpy.ndarray
        The array to be converted

    Returns
    -------
    System.Array

    Warning
    -------
    ``complex64`` and ``complex128`` arrays are converted to ``float32``
    and ``float64`` arrays respectively with shape ``[m,n,...] -> [m,n,...,2]``

    """
    dims = npArray.shape
    dtype = npArray.dtype

    # For complex arrays, we must make a view of the array as its corresponding 
    # float type as if it's (real, imag)
    if dtype == np.complex64:
        dtype = np.dtype(np.float32)
        dims += (2,)
        npArray = npArray.view(dtype).reshape(dims)
    elif dtype == np.complex128:
        dtype = np.dtype(np.float64)
        dims += (2,)
        npArray = npArray.view(dtype).reshape(dims)

    if not npArray.flags.c_contiguous or not npArray.flags.aligned:
        npArray = np.ascontiguousarray(npArray)
    assert npArray.flags.c_contiguous

    try:
        netArray = Array.CreateInstance(_MAP_NP_NET[dtype], *dims)
    except KeyError:
        raise NotImplementedError(f'asNetArray does not yet support dtype {dtype}')

    try: # Memmove 
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__['data'][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated: 
            destHandle.Free()
    return netArray
