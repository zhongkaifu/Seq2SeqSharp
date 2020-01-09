using System;
using System.Reflection;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
    public class PrecompileAttribute : Attribute
    {
        public PrecompileAttribute()
        {
        }
    }

    public interface IPrecompilable
    {
        void Precompile(CudaCompiler compiler);
    }

    public static class PrecompileHelper
    {
        public static void PrecompileAllFields(object instance, CudaCompiler compiler)
        {
            Type type = instance.GetType();

            foreach (FieldInfo field in type.GetFields())
            {
                if (typeof(IPrecompilable).IsAssignableFrom(field.FieldType))
                {
                    IPrecompilable precompilableField = (IPrecompilable)field.GetValue(instance);
                    Console.WriteLine("Compiling field " + field.Name);
                    precompilableField.Precompile(compiler);
                }
            }
        }
    }
}
