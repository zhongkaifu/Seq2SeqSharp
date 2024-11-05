using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;

namespace AdvUtils
{
	public class ArgParser
	{
        object m_o;
        List<ArgField> m_arrayArgs;

        public static void UpdateFieldValue(object obj, string fieldName, string newValue)
        {
            // Get the Type of the object
            Type objType = obj.GetType();

            // Get the FieldInfo for the specified field name
            FieldInfo fieldInfo = objType.GetField(fieldName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            // If the field is found, set the new value
            if (fieldInfo != null)
            {
                fieldInfo.SetValue(obj, newValue);
            }
            else
            {
                Console.WriteLine($"Field '{fieldName}' not found in the class.");
            }
        }

        public ArgParser(string[] args, object o)
        {
            m_o = o;
            m_arrayArgs = new List<ArgField>();
            Type typeArgAttr = typeof(Arg);
            Type t = o.GetType();
            foreach (FieldInfo fi in t.GetFields(BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance))
            {
                foreach (Arg arg in fi.GetCustomAttributes(typeArgAttr, true))
                {
                    m_arrayArgs.Add(new ArgField(o, fi, arg));
                }
            }

            RewriteSettings(args, o);
        }

        public void RewriteSettings(string[] args, object o)
        {
            try
            {
                for (int i = 0; i < args.Length; i++)
                {
                    if (args[i].StartsWith("-"))
                    {
                        string strArgName = args[i].Substring(1);
                        string strArgValue = args[i + 1];

                        ArgField? intarg = GetArgByName(strArgName);
                        if (intarg == null)
                        {
                            throw new ArgumentException($"{strArgName} is not a valid parameter");
                        }

                        intarg.Set(strArgValue);

                        Console.WriteLine($"Rewrite field '{strArgName}' value.");
                        UpdateFieldValue(o, strArgName, strArgValue);

                        i++;
                    }
                }

                foreach (ArgField a in m_arrayArgs)
                    a.Validate();
            }
            catch (Exception err)
            {
                Console.Error.WriteLine(err.Message);
                Usage();
            }
        }

        ArgField? GetArgByName(string name)
		{
			foreach (ArgField a in m_arrayArgs)
				if (a.Arg.Name.ToLower() == name.ToLower())
					return a;
			return null;
		}

		public void Usage()
		{
			string strAppName = Process.GetCurrentProcess().ProcessName;
            Console.Error.WriteLine("Usage: {0} [parameters...]", strAppName);

            foreach (var item in m_arrayArgs)
            {
                Console.Error.WriteLine($"\t[-{item.Arg.Name}: {item.Arg.Title}]");
            }

            System.Environment.Exit(-1);
        }

	}
}

