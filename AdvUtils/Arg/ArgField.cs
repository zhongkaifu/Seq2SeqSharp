using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace AdvUtils
{
    class ArgField
    {
        public ArgField(object o, FieldInfo fi, Arg a)
        {
            m_o = o;
            m_fi = fi;
            m_a = a;
            m_fSet = false;
        }

        public void Set(string val)
        {
            try
            {
                if (m_fi.FieldType == typeof(string))
                {
                    m_fi.SetValue(m_o, val);
                }
                else
                {
                    Type argumentType = m_fi.FieldType.IsGenericType && m_fi.FieldType.GetGenericTypeDefinition() == typeof(Nullable<>) ?
                        m_fi.FieldType.GenericTypeArguments[0] : m_fi.FieldType;

                    MethodInfo mi = argumentType.GetMethod("Parse", new Type[] { typeof(string) });
                    if (mi != null)
                    {
                        object oValue = mi.Invoke(null, new object[] { val });
                        m_fi.SetValue(m_o, oValue);
                    }
                    else if (argumentType.IsEnum)
                    {
                        object oValue = Enum.Parse(m_fi.FieldType, val);
                        m_fi.SetValue(m_o, oValue);
                    }
                }
                m_fSet = true;
            }
            catch (Exception err)
            {
                throw new ArgumentException($"Failed to set value of '{m_a.ToString()}', Error: '{err.Message}', Call Stack: '{err.StackTrace}'");
            }
        }
        public void Validate()
        {
            if (!m_a.Optional && !m_fSet)
                throw new ArgumentException($"Failed to specify value for required {m_a.ToString()}");
        }
        public Arg Arg { get { return m_a; } }

        object m_o;
        FieldInfo m_fi;
        Arg m_a;
        bool m_fSet;
    }
}
