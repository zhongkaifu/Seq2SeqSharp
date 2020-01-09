using System.Collections.Generic;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    public class KernelConfig
    {
        private readonly SortedDictionary<string, string> values = new SortedDictionary<string, string>();


        public KernelConfig()
        {
        }

        public IEnumerable<string> Keys => values.Keys;

        public IEnumerable<KeyValuePair<string, string>> AllValues()
        {
            return values;
        }

        public override bool Equals(object obj)
        {
            KernelConfig o = obj as KernelConfig;
            if (o == null)
            {
                return false;
            }

            if (values.Count != o.values.Count)
            {
                return false;
            }

            foreach (KeyValuePair<string, string> kvp in values)
            {
                if (values.TryGetValue(kvp.Key, out string oValue))
                {
                    if (!kvp.Value.Equals(oValue))
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        public override int GetHashCode()
        {
            int result = 0;
            foreach (KeyValuePair<string, string> kvp in values)
            {
                result ^= kvp.Key.GetHashCode();
                result ^= kvp.Value.GetHashCode();
            }
            return result;
        }

        public bool ContainsKey(string name)
        {
            return values.ContainsKey(name);
        }

        public void Set(string name, string value)
        {
            values[name] = value;
        }

        public string ApplyToTemplate(string templateCode)
        {
            StringBuilder fullCode = new StringBuilder();
            foreach (KeyValuePair<string, string> item in values)
            {
                fullCode.AppendFormat("#define {0} {1}\n", item.Key, item.Value);
            }
            fullCode.AppendLine(templateCode);
            return fullCode.ToString();
        }
    }
}
