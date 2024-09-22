from System import Type
#from pythonnet import load
#load("coreclr")
import clr
import sys
import os

def InitDevices(deviceType, deviceIds):

    dir = os.path.dirname(sys.modules["Seq2SeqSharp"].__file__)
    path = os.path.join(dir, "Seq2SeqSharp.dll")

    print(path)

    lib = clr.AddReference(path)
    my_type = lib.GetType('Seq2SeqSharp.TensorAllocator')
    method = my_type.GetMethod('InitDevices')

    # RetType is void in my case, so None works
    RetType = None
    # parameters passed to the functions need to be a list
    method.Invoke(RetType, [deviceType, deviceIds, 0.95, None])
