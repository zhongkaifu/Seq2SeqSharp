using System;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    public class CudaIncludeAttribute : Attribute
    {
        public string FieldName { get; private set; }
        public string IncludeName { get; private set; }

        public CudaIncludeAttribute(string fieldName, string includeName)
        {
            FieldName = fieldName;
            IncludeName = includeName;
        }
    }
}
