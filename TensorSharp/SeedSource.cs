using System;

namespace TensorSharp
{
    public class SeedSource
    {
        private Random rng;

        public SeedSource(int seed)
        {
            rng = new Random();
        }

        public int NextSeed()
        {
            return rng.Next();
        }
    }
}
