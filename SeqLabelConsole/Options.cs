using AdvUtils;

namespace SeqLabelConsole
{
    internal class Options : Seq2SeqSharp.Applications.Options
    {
        [Arg("Task name. It could be Train, Valid, Test, VisualizeNetwork or Help", "TaskName")]
        public string TaskName = "Help";

        [Arg("The vector size of encoded source word.", "WordVectorSize")]
        public int WordVectorSize = 128;

        [Arg("The test result file.", "OutputTestFile")]
        public string OutputTestFile = null;


        [Arg("Beam search size. Default is 1", "BeamSearch")]
        public int BeamSearch = 1;

        [Arg("Maxmium sentence length", "MaxSentLength")]
        public int MaxSentLength = 128;
    }
}
