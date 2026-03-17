namespace Seq2SeqSharp.Utils
{
    public enum ProcessorTypeEnums
    {
        CUDA,
        CPU,
        CPU_MKL,
        GGML_METAL
    }

    public static class ProcessorTypeEnumExtensions
    {
        public static bool IsCuda(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.CUDA;
        }

        public static bool IsCpu(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.CPU || processorType == ProcessorTypeEnums.CPU_MKL;
        }

        public static bool IsGGML(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.GGML_METAL;
        }

        public static bool UsesAccelerator(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.CUDA || processorType == ProcessorTypeEnums.GGML_METAL;
        }
    }

    public enum LearningRateTypeEnums
    {
        Decay,
        CosineDecay
    }
}
