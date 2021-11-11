using System;

namespace SeqWebApps
{
    [Serializable]
    public class TextGenerationModel
    {
        ///<summary>
        /// Gets or sets Output.
        ///</summary>
        public string Output { get; set; }

        ///<summary>
        /// Gets or sets DateTime.
        ///</summary>
        public string DateTime { get; set; }
    }
}