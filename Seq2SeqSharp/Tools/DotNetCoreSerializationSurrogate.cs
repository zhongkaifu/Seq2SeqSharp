using AdvUtils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.Serialization;

namespace Seq2SeqSharp.Tools
{
    internal static class DotNetCoreSerializationSurrogate
    {
        internal static ISurrogateSelector CreateSurrogateSelector()
        {
            var surrogateSelector = new SurrogateSelector();

            ConcurrentDictionarySerializationSurrogate<int, string>.CreateAndRegister(surrogateSelector);
            ConcurrentDictionarySerializationSurrogate<string, int>.CreateAndRegister(surrogateSelector);

            return surrogateSelector;
        }

        private sealed class ConcurrentDictionarySerializationSurrogate<TKey, TValue> : ISerializationSurrogate
        {
            public static void CreateAndRegister(SurrogateSelector surrogateSelector)
            {
                surrogateSelector.AddSurrogate(
                    typeof(ConcurrentDictionary<TKey, TValue>),
                    new StreamingContext(StreamingContextStates.All),
                    new ConcurrentDictionarySerializationSurrogate<TKey, TValue>()
                );
            }

            public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
            {
                throw new NotImplementedException("DictionarySerializationSurrogate.GetObjectData");
            }

            public object SetObjectData(
                object obj,
                SerializationInfo info,
                StreamingContext context,
                ISurrogateSelector selector)
            {
                Logger.WriteLine($"[SetObjectData] called: {obj}");

                var values = info.GetValue("m_serializationArray", typeof(KeyValuePair<TKey, TValue>[])) as KeyValuePair<TKey, TValue>[];

                // NOTE: Construct a new ConcurrentDictionary directly because the serialization framework doesn't
                // initialize obj in the expected way.
                return new ConcurrentDictionary<TKey, TValue>(values);
            }
        }
    }
}
