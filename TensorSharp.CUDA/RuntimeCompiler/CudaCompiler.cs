using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    [Serializable]
    public class CudaCompiler
    {
        private readonly Dictionary<string, string> includes = new Dictionary<string, string>();
        private readonly KernelDiskCache diskCache;

        public CudaCompiler(KernelDiskCache diskCache)
        {
            this.diskCache = diskCache;
            RegisterAttributeHeaders(Assembly.GetExecutingAssembly());
        }

        public byte[] CompileToPtx(string code, params string[] prependIncludes)
        {
            // We manually prepend include files here, so that the header content forms part of the hash of the source
            // code. This means that changes to headers will correctly trigger a recompile.
            StringBuilder finalCode = new StringBuilder();
            foreach (string includeName in prependIncludes)
            {
                finalCode.Append(includes[includeName]).Append('\n');
            }
            finalCode.Append(code);
            string finalCodeString = finalCode.ToString();

            return diskCache.Get(finalCodeString, DoCompile);
        }

        private byte[] DoCompile(string fullSource)
        {
            ManagedCuda.NVRTC.CudaRuntimeCompiler rtc = new ManagedCuda.NVRTC.CudaRuntimeCompiler(fullSource, null);

            try
            {
                rtc.Compile(new string[] { "--use_fast_math" });
            }
            catch
            {
                throw new ApplicationException("Error compiling CUDA code: " + rtc.GetLogAsString());
            }

            return rtc.GetPTX();
        }

        public void RegisterHeader(string name, string content)
        {
            includes.Add(name, content);
        }


        private void RegisterAttributeHeaders(Assembly assembly)
        {
            foreach (Tuple<Type, IEnumerable<CudaIncludeAttribute>> applyType in assembly.TypesWithAttribute<CudaIncludeAttribute>(false))
            {
                foreach (CudaIncludeAttribute attribute in applyType.Item2)
                {
                    Tuple<string, string> info = HeaderInfoFromAttribute(applyType.Item1, attribute);
                    RegisterHeader(info.Item1, info.Item2);
                }
            }
        }

        private Tuple<string, string> HeaderInfoFromAttribute(Type containingType, CudaIncludeAttribute attribute)
        {
            FieldInfo field = containingType.GetField(attribute.FieldName, BindingFlags.Public | BindingFlags.Static);
            string content = (string)field.GetValue(null);
            return Tuple.Create(attribute.IncludeName, content);
        }
    }
}
