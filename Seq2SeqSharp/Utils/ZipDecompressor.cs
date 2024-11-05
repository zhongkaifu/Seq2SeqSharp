using ICSharpCode.SharpZipLib.Zip;
using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Utils
{
    public class ZipDecompressor : IDisposable
    {
        private readonly string _tempFilePath;
        private MemoryMappedFile _memoryMappedFile;

        public ZipDecompressor(string zipFilePath, string password)
        {
            _tempFilePath = Path.GetTempFileName();

            // Register event handlers for process exit and unhandled exception
            AppDomain.CurrentDomain.ProcessExit += OnProcessExit;
            AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;

            // Decompress the ZIP file to the temp file
            DecompressZipToTempFile(zipFilePath, password);

            // Create a memory-mapped file from the decompressed temp file
            _memoryMappedFile = MemoryMappedFile.CreateFromFile(_tempFilePath, FileMode.Open, null);
        }

        public MemoryMappedViewStream GetMemoryMappedViewStream()
        {
            return _memoryMappedFile?.CreateViewStream();
        }

        private void DecompressZipToTempFile(string zipFilePath, string password)
        {
            using (FileStream fs = File.OpenRead(zipFilePath))
            using (ZipInputStream zipStream = new ZipInputStream(fs))
            {
                zipStream.Password = password;

                using (FileStream tempFileStream = new FileStream(_tempFilePath, FileMode.Create, FileAccess.Write, FileShare.Delete, 4096))
                {
                    byte[] buffer = new byte[8192];
                    ZipEntry entry;
                    while ((entry = zipStream.GetNextEntry()) != null)
                    {
                        int size;
                        while ((size = zipStream.Read(buffer, 0, buffer.Length)) > 0)
                        {
                            tempFileStream.Write(buffer, 0, size);
                        }
                    }
                }
            }
        }

        private void OnProcessExit(object sender, EventArgs e)
        {
            // Cleanup when the process exits
            Dispose();
        }

        private void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            // Cleanup when an unhandled exception occurs
            Dispose();
        }

        public void Dispose()
        {
            // Cleanup memory-mapped file and delete the temp file
            _memoryMappedFile?.Dispose();
            _memoryMappedFile = null;

            if (File.Exists(_tempFilePath))
            {
                try
                {
                    File.Delete(_tempFilePath);
                    Console.WriteLine("Temporary file deleted successfully.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Failed to delete temporary file: " + ex.Message);
                }
            }

            // Unregister event handlers to avoid memory leaks
            AppDomain.CurrentDomain.ProcessExit -= OnProcessExit;
            AppDomain.CurrentDomain.UnhandledException -= OnUnhandledException;
        }

        //static void Main(string[] args)
        //{
        //    string zipFilePath = "path/to/your/password-protected.zip";
        //    string password = "your_password";

        //    using (var decompressor = new ZipDecompressor(zipFilePath, password))
        //    {
        //        using (MemoryMappedViewStream memoryMappedViewStream = decompressor.GetMemoryMappedViewStream())
        //        {
        //            // Use the memory-mapped view stream
        //            Console.WriteLine("Decompression completed, stream length: " + memoryMappedViewStream.Length);

        //            // Perform operations with the stream here...
        //        }
        //    }

        //    // The temp file will be automatically deleted when the ZipDecompressor is disposed.
        //}
    }
}
