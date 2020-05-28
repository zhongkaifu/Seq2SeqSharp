using AdvUtils;
using Seq2SeqSharp.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Utils
{
    public class MaskUtils
    {
        public static IWeightTensor BuildPadMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int deviceId)
        {
            float[] buf = new float[originalLengths.Count * paddedLength];
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e9f;
            }

            for (int i = 0; i < originalLengths.Count; i++)
            {
                for (int j = 0; j < originalLengths[i]; j++)
                {
                    buf[i * paddedLength + j] = 0.0f;
                }
            }

            WeightTensor tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength }, 0.0f, deviceId, $"Mask_{deviceId}", isTrainable: false);
            tensor.SetWeightArray(buf);

            return tensor;
        }


        public static IWeightTensor BuildPadDimMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int dim, int deviceId)
        {
            float[] buf = new float[originalLengths.Count * paddedLength * dim];
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e9f;
            }

            for (int k = 0; k < originalLengths.Count; k++)
            {
                for (int i = 0; i < originalLengths[k]; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        buf[k * (paddedLength * dim) + i * dim + j] = 0.0f;
                    }
                }
            }

            WeightTensor tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, dim }, 0.0f, deviceId, $"TriMask_{deviceId}", isTrainable: false);
            tensor.SetWeightArray(buf);

            return tensor;
        }


        public static IWeightTensor BuildPadSelfMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int deviceId)
        {
            float[] buf = new float[originalLengths.Count * paddedLength * paddedLength];
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e38f;
            }

            for (int k = 0; k < originalLengths.Count; k++)
            {
                for (int i = 0; i < originalLengths[k]; i++)
                {
                    for (int j = 0; j < originalLengths[k]; j++)
                    {
                        buf[k * (paddedLength * paddedLength) + i * paddedLength + j] = 0.0f;
                    }
                }
            }

            WeightTensor tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, paddedLength }, 0.0f, deviceId, $"TriMask_{deviceId}", isTrainable: false);
            tensor.SetWeightArray(buf);

            return tensor;
        }

        public static IWeightTensor BuildPadSelfTriMask(IComputeGraph g, int paddedLength, List<int> originalLengths, int deviceId)
        {
            float[] buf = new float[originalLengths.Count * paddedLength * paddedLength];
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e38f;
            }

            for (int k = 0; k < originalLengths.Count; k++)
            {
                for (int i = 0; i < originalLengths[k]; i++)
                {
                    for (int j = 0; j < originalLengths[k]; j++)
                    {
                        if (i >= j)
                        {
                            buf[k * (paddedLength * paddedLength) + i * paddedLength + j] = 0.0f;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            WeightTensor tensor = new WeightTensor(new long[] { originalLengths.Count, paddedLength, paddedLength }, 0.0f, deviceId, $"TriMask_{deviceId}", isTrainable: false);
            tensor.SetWeightArray(buf);

            return tensor;
        }


        public static IWeightTensor BuildSrcTgtMask(IComputeGraph g, int srcPaddedLength, int tgtPaddedLength, List<int> tgtOriginalLengths, int deviceId)
        {
            float[] buf = new float[tgtOriginalLengths.Count * tgtPaddedLength * srcPaddedLength];
            for (int i = 0; i < buf.Length; i++)
            {
                buf[i] = -1e38f;
            }

            for (int k = 0; k < tgtOriginalLengths.Count; k++) // batch size
            {
                for (int i = 0; i < tgtOriginalLengths[k]; i++)
                {
                    for (int j = 0; j < srcPaddedLength; j++)
                    {
                        buf[k * (tgtPaddedLength * srcPaddedLength) + i * srcPaddedLength + j] = 0.0f;
                    }
                }
            }

            WeightTensor tensor = new WeightTensor(new long[] { tgtOriginalLengths.Count, tgtPaddedLength, srcPaddedLength }, 0.0f, deviceId, $"SrcTgtMask_{deviceId}", isTrainable: false);
            tensor.SetWeightArray(buf);

            return tensor;
        }
    }
}
