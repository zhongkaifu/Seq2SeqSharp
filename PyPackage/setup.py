from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Seq2SeqSharp',
    version='2.7.4',
    author='Zhongkai Fu',
    author_email='fuzhongkai@gmail.com',
    description='Seq2SeqSharp is a tensor based fast & flexible encoder-decoder deep neural network framework written by .NET (C#). It has many highlighted features, such as automatic differentiation, many different types of encoders/decoders(Transformer, LSTM, BiLSTM and so on), multi-GPUs supported, cross-platforms (Windows, Linux, x86, x64, ARM) and so on.',
    url='https://github.com/zhongkaifu/Seq2SeqSharp',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    packages=['Seq2SeqSharp'],
    install_requires=['pythonnet>=3.0.1'],
    data_files=[
        ('Seq2SeqSharp', ['Seq2SeqSharp/AdvUtils.dll','Seq2SeqSharp/AdvUtils.pdb','Seq2SeqSharp/CudaBlas.dll','Seq2SeqSharp/CudaBlas.pdb','Seq2SeqSharp/CudaRand.dll','Seq2SeqSharp/CudaRand.pdb','Seq2SeqSharp/ManagedCuda.dll','Seq2SeqSharp/ManagedCuda.pdb','Seq2SeqSharp/Microsoft.Extensions.Caching.Abstractions.dll','Seq2SeqSharp/Microsoft.Extensions.Caching.Memory.dll','Seq2SeqSharp/Microsoft.Extensions.DependencyInjection.Abstractions.dll','Seq2SeqSharp/Microsoft.Extensions.Logging.Abstractions.dll','Seq2SeqSharp/Microsoft.Extensions.Options.dll','Seq2SeqSharp/Microsoft.Extensions.Primitives.dll','Seq2SeqSharp/Newtonsoft.Json.dll','Seq2SeqSharp/NVRTC.dll','Seq2SeqSharp/NVRTC.pdb','Seq2SeqSharp/protobuf-net.Core.dll','Seq2SeqSharp/protobuf-net.dll','Seq2SeqSharp/Seq2SeqSharp.dll','Seq2SeqSharp/Seq2SeqSharp.dll.config','Seq2SeqSharp/Seq2SeqSharp.pdb','Seq2SeqSharp/TensorSharp.CUDA.dll','Seq2SeqSharp/TensorSharp.CUDA.dll.config','Seq2SeqSharp/TensorSharp.CUDA.pdb','Seq2SeqSharp/TensorSharp.dll','Seq2SeqSharp/TensorSharp.pdb'])],
    include_package_data=True
)


