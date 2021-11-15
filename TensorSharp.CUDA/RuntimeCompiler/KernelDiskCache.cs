using AdvUtils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{

    [Serializable]
    public class KernelDiskCache
    {
        private readonly string cacheDir;
        private readonly Dictionary<string, byte[]> memoryCachedKernels = new Dictionary<string, byte[]>();


        public KernelDiskCache(string cacheDir)
        {
            this.cacheDir = cacheDir;
            if (!System.IO.Directory.Exists(cacheDir))
            {
                System.IO.Directory.CreateDirectory(cacheDir);
            }
        }

        /// <summary>
        /// Deletes all kernels from disk if they are not currently loaded into memory. Calling this after
        /// calling TSCudaContext.Precompile() will delete any cached .ptx files that are no longer needed
        /// </summary>
        public void CleanUnused()
        {
            foreach (string file in Directory.GetFiles(cacheDir))
            {
                string key = KeyFromFilePath(file);
                if (!memoryCachedKernels.ContainsKey(key))
                {
                    File.Delete(file);
                }
            }
        }

        public byte[] Get(string fullSourceCode, Func<string, byte[]> compile)
        {
            string key = KeyFromSource(fullSourceCode);
            if (memoryCachedKernels.TryGetValue(key, out byte[] ptx))
            {
                return ptx;
            }
            else if (TryGetFromFile(key, out ptx))
            {
                memoryCachedKernels.Add(key, ptx);
                return ptx;
            }
            else
            {
                WriteCudaCppToFile(key, fullSourceCode);

                ptx = compile(fullSourceCode);
                memoryCachedKernels.Add(key, ptx);
                WriteToFile(key, ptx);

                return ptx;
            }
        }


        private void WriteToFile(string key, byte[] ptx)
        {
            string filePath = FilePathFromKey(key);

            Logger.WriteLine($"Writing PTX code to '{filePath}'");
            System.IO.File.WriteAllBytes(filePath, ptx);
        }

        private void WriteCudaCppToFile(string key, string sourceCode)
        {           
            string filePath = FilePathFromKey(key) + ".cu";

            Logger.WriteLine($"Writing cuda source code to '{filePath}'");
            System.IO.File.WriteAllText(filePath, sourceCode);
        }

        private bool TryGetFromFile(string key, out byte[] ptx)
        {
            string filePath = FilePathFromKey(key);
            if (!System.IO.File.Exists(filePath))
            {
                ptx = null;
                return false;
            }

            ptx = System.IO.File.ReadAllBytes(filePath);
            return true;
        }

        private string FilePathFromKey(string key)
        {
            return System.IO.Path.Combine(cacheDir, key + ".ptx");
        }

        private string KeyFromFilePath(string filepath)
        {
            string[] fileExts = new string[] { ".ptx", ".cu" };

            foreach (string ext in fileExts)
            {
                if (filepath.EndsWith(ext, StringComparison.InvariantCultureIgnoreCase))
                {
                    filepath = filepath.Substring(0, filepath.Length - ext.Length);
                }
            }

     //       return filepath;
           

            return Path.GetFileNameWithoutExtension(filepath);
        }

        private static string KeyFromSource(string fullSource)
        {
            string fullKey = fullSource.Length.ToString() + fullSource;            
            using (var sha1 = SHA1.Create())
            {
                return BitConverter.ToString(sha1.ComputeHash(Encoding.UTF8.GetBytes(fullKey)))
                    .Replace("-", "");
            }
        }
    }
}
