using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace AdvUtils
{

    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public class Arg : Attribute
    {
        public Arg(string title, string name, bool optional = true)
        {
            Title = title;
            Name = name;
            Optional = optional;
        }


        public override string ToString()
        {
            return "argument " + Name + " (" + Title + ")";
        }

        public string UsageLineText()
        {
            string s = Optional ? "[" : "";
            s += ($"-{Name}: {Title}");           
            if (Optional) s += "]";

            return s;
        }

        public string Title;
        public bool Optional;
        public string Name;
    }
}
