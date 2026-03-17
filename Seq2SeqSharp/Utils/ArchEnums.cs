namespace Seq2SeqSharp.Utils
{
    public enum ProcessorTypeEnums
    {
        GPU,
        CPU,
        CPU_MKL,
        GGML
    }

    public static class ProcessorTypeEnumExtensions
    {
        public static bool IsCuda(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.GPU;
        }

        public static bool IsCpu(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.CPU || processorType == ProcessorTypeEnums.CPU_MKL;
        }

        public static bool IsGGML(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.GGML;
        }

        public static bool UsesAccelerator(this ProcessorTypeEnums processorType)
        {
            return processorType == ProcessorTypeEnums.GPU || processorType == ProcessorTypeEnums.GGML;
        }
    }

    public enum LearningRateTypeEnums
    {
        Decay,
        CosineDecay
    }
}
