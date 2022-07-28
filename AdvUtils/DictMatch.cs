/*
 * Project: DictMatch
 * Dictmatch is a high performance multi-mode string match lib. 
 * it is used to match substrings which contented in dictionary from string user provides
 * Author: Zhongkai Fu
 * Email: fuzhongkai@gmail.com
 */
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace AdvUtils
{
    public class Lemma
    {
        public uint len;
        public string? strProp;
    };

    public class dm_entry_t
    {
        public int value;
        public int lemma_pos;
        public int suffix_pos;
    };

    //存放每个节点的子树的指针列表（即后继块）
    public class sufentry
    {
        public int hashsize; //该后继块的hash表大小
        public int backsepos; //指向其属主dentry节点
        public int[]? hashList; //存放子树指针的hash表
    }

    public class DictMatch
    {
        private const int DM_DENTRY_NULL = -1;
        private const int DM_DENTRY_FIRST = -2;
        private const int DM_SUFENTRY_NULL = -1;
        private const int DM_LEMMA_NULL = -1;
        private const int DM_COMMON_NULL = -1;

        //匹配类型
        public const int DM_OUT_ALL = 0; //全部匹配
        public const int DM_OUT_FMM = 1; //最大正向匹配

        private const int DM_DEFAULT_SEBUFSIZE = 1024000;


        private List<dm_entry_t> dentry; //  存放树的每个节点
        private List<sufentry> seinfo;
        private int sebufsize;
        private List<Lemma> lmlist; // 存放完成匹配对应的某模式

        //指向了虚根节点所引出的每棵树的指针列表，也就是整个Trie树的入口。
        private int entrance;

        public DictMatch()
        {
            dentry = new List<dm_entry_t>();
            seinfo = new List<sufentry>();
            lmlist = new List<Lemma>();
        }

        public void LoadDictFromBinary(string strFileName)
        {
            StreamReader sr = new StreamReader(strFileName);
            BinaryReader br = new BinaryReader(sr.BaseStream);

            dentry = new List<dm_entry_t>();
            seinfo = new List<sufentry>();
            lmlist = new List<Lemma>();

            entrance = br.ReadInt32();

            int dentryCount = br.ReadInt32();
            for (int i = 0; i < dentryCount; i++)
            {
                dm_entry_t entry = new dm_entry_t();
                entry.lemma_pos = br.ReadInt32();
                entry.suffix_pos = br.ReadInt32();
                entry.value = br.ReadInt32();

                dentry.Add(entry);
            }

            int seinfoCount = br.ReadInt32();
            for (int i = 0; i < seinfoCount; i++)
            {
                sufentry suf = new sufentry();
                suf.backsepos = br.ReadInt32();
                suf.hashsize = br.ReadInt32();

                suf.hashList = new int[suf.hashsize];
                for (int j = 0; j < suf.hashsize; j++)
                {
                    suf.hashList[j] = br.ReadInt32();
                }

                seinfo.Add(suf);
            }

            sebufsize = br.ReadInt32();

            int lmlistCount = br.ReadInt32();
            for (int i = 0; i < lmlistCount; i++)
            {
                Lemma lemma = new Lemma();
                lemma.len = br.ReadUInt32();
                lemma.strProp = br.ReadString();

                lmlist.Add(lemma);
            }

            br.Close();
        }

        public void ConvertDictFromRawTextToBinary(string strRawFileName, string strDestFileName)
        {
            LoadDictFromRawText(strRawFileName);

            StreamWriter sw = new StreamWriter(strDestFileName);
            BinaryWriter bw = new BinaryWriter(sw.BaseStream);

            bw.Write(entrance);

            bw.Write(dentry.Count);
            foreach (dm_entry_t item in dentry)
            {
                bw.Write(item.lemma_pos);
                bw.Write(item.suffix_pos);
                bw.Write(item.value);
            }

            bw.Write(seinfo.Count);
            foreach (sufentry item in seinfo)
            {
                bw.Write(item.backsepos);
                bw.Write(item.hashsize);

                if (item.hashList == null)
                {
                    throw new NullReferenceException($"Null hash list in the item.");
                }

                foreach (int id in item.hashList)
                {
                    bw.Write(id);
                }
            }

            bw.Write(sebufsize);

            bw.Write(lmlist.Count);
            foreach (Lemma item in lmlist)
            {
                bw.Write(item.len);
                bw.Write(item.strProp);
            }



            bw.Close();

        }


        private void Init()
        {
            dentry = new List<dm_entry_t>();
            seinfo = new List<sufentry>();
            sebufsize = DM_DEFAULT_SEBUFSIZE;
            lmlist = new List<Lemma>();

            entrance = 0;
            sufentry s = InitSufentry(1, DM_DENTRY_FIRST);
            seinfo.Add(s);
        }

        public void LoadDictFromRawText(string fullpath)
        {
            Init();

            StreamReader sr = new StreamReader(fullpath);
            while (sr.EndOfStream == false)
            {
                string? strLine = sr.ReadLine();
                if (strLine == null)
                {
                    break;
                }

                string[]? items = strLine.Split(new char[] { '\t' });

                string strTerm;
                string strProp = "";

                if (items.Length < 1)
                {
                    throw new System.Exception("dict file is invalidated");
                }
                strTerm = items[0]; //.Trim();
                //Ignore empty line
                if (strTerm.Length == 0)
                {
                    continue;
                }

                if (items.Length == 2)
                {
                    strProp = items[1];
                }

                if (AddLemma(strTerm, strProp) < 0)
                {
                    throw new System.Exception("dm_dict_load error!");
                }
            }
            sr.Close();
        }

        //字典词条匹配
        //inbuf : 输入字符串
        //dm_r : 字典词条属性
        //offsetList : 字典词条在字符串中的偏移量（如果输入字符串中包含多个相同的字典词条，那么分别记录下他们不同的偏移量）
        //opertype : 匹配模式
        public int Search(string inbuf, ref List<Lemma> dm_r, ref List<int> offsetList, int opertype)
        {
            int bpos = 0;
            int pos = 0;
            int nextpos = 0;
            int nde = 0, lde = DM_DENTRY_FIRST;
            int lmpos = DM_LEMMA_NULL;
            int slemma = DM_LEMMA_NULL;

            dm_r.Clear();
            offsetList.Clear();
            if (opertype == DM_OUT_FMM)
            {
                while (pos < inbuf.Length)
                {
                    bpos = pos;
                    nextpos = pos + 1;
                    while (pos < inbuf.Length)
                    {
                        nde = SeekEntry(lde, inbuf[pos]);
                        if (nde == DM_DENTRY_NULL)
                        {
                            break;
                        }
                        lmpos = dentry[nde].lemma_pos;
                        if (lmpos != DM_LEMMA_NULL)
                        {
                            slemma = lmpos;
                            nextpos = pos + 1;
                        }

                        lde = nde;
                        pos++;
                    }

                    if (slemma != DM_LEMMA_NULL)
                    {
                        offsetList.Add(bpos);
                        dm_r.Add(lmlist[slemma]);
                    }

                    lde = DM_DENTRY_FIRST;
                    slemma = DM_LEMMA_NULL;
                    pos = nextpos;
                }
            }
            else if (opertype == DM_OUT_ALL)
            {
                while (pos < inbuf.Length)
                {
                    bpos = pos;
                    nextpos = pos + 1;

                    while (pos < inbuf.Length)
                    {
                        nde = SeekEntry(lde, inbuf[pos]);
                        if (nde == DM_DENTRY_NULL)
                        {
                            break;
                        }

                        lmpos = dentry[nde].lemma_pos;
                        if (lmpos != DM_LEMMA_NULL)
                        {
                            offsetList.Add(bpos);
                            dm_r.Add(lmlist[lmpos]);
                        }
                        lde = nde;
                        pos++;
                    }

                    lde = DM_DENTRY_FIRST;
                    pos = nextpos;
                }
            }

            return 0;
        }

        private int SeekEntry(int lde, int value)
        {
            int sufpos;
            int nde;
            int hsize;
            int hpos;

            if (lde == DM_DENTRY_FIRST)
            {
                sufpos = entrance;
            }
            else
            {
                sufpos = dentry[lde].suffix_pos;
            }
            if (sufpos == DM_SUFENTRY_NULL)
            {
                return DM_DENTRY_NULL;
            }

            if (seinfo == null)
            {
                throw new NullReferenceException("seinfo is null.");
            }

            hsize = seinfo[sufpos].hashsize;
            hpos = value % hsize;
            if (((nde = seinfo[sufpos].hashList[hpos]) == DM_DENTRY_NULL)
               || (dentry[nde].value != value))
            {
                return DM_DENTRY_NULL;
            }
            else
            {
                return nde;
            }
        }

        private int AddLemma(string strTerm, string strProp)
        {
            int curpos = 0;
            int last_depos = DM_DENTRY_FIRST;
            int cur_depos = DM_COMMON_NULL;
            int value = 0;
            int lmpos = DM_COMMON_NULL;

            //Check if lemma has already in the dictionary
            if ((lmpos = SeekLemma(strTerm)) == DM_LEMMA_NULL)
            {
                //new lemma, insert it into the dictionary
                while (curpos < strTerm.Length)
                {
                    value = strTerm[curpos];
                    curpos++;
                    if (InsertDentry(last_depos, value, ref cur_depos) < 0)
                    {
                        return -1;
                    }
                    last_depos = cur_depos;
                }

                dentry[cur_depos].lemma_pos = lmlist.Count;

                Lemma lm = new Lemma();
                lm.strProp = strProp;
                lm.len = (uint)strTerm.Length;

                lmlist.Add(lm);

                return 1;
            }
            return 0;
        }

        private int InsertDentry(int lastpos, int value, ref int curpos)
        {
            int tmpdepos;
            int curdepos;
            int sufpos;
            int hsize;
            int hpos;

            if (lastpos != DM_DENTRY_FIRST)
            {
                sufpos = dentry[lastpos].suffix_pos;
            }
            else
            {
                sufpos = entrance;
            }

            if (sufpos == DM_SUFENTRY_NULL)
            {
                if (seinfo.Count > sebufsize)
                {
                    if (ResizeInfo() < 0)
                    {
                        return -1;
                    }
                }

                dentry[lastpos].suffix_pos = seinfo.Count;
                sufentry s = InitSufentry(1, lastpos);
                seinfo.Add(s);
                sufpos = dentry[lastpos].suffix_pos;
            }


            hsize = seinfo[sufpos].hashsize;
            hpos = value % hsize;
            tmpdepos = seinfo[sufpos].hashList[hpos];
            if ((tmpdepos != DM_DENTRY_NULL) && (dentry[tmpdepos].value == value))
            {
                curpos = tmpdepos;
                return 0;
            }
            else
            {
                dm_entry_t det = new dm_entry_t();
                det.value = value;
                det.lemma_pos = DM_LEMMA_NULL;
                det.suffix_pos = DM_SUFENTRY_NULL;

                curdepos = dentry.Count;
                dentry.Add(det);

                if (tmpdepos == DM_DENTRY_NULL)
                {
                    seinfo[sufpos].hashList[hpos] = curdepos;
                    curpos = curdepos;
                    return 1;
                }
                else
                {
                    int newhash;
                    for (newhash = hsize + 1; ; newhash++)
                    {
                        int conflict = 0;
                        if (seinfo.Count > sebufsize)
                        {
                            if (ResizeInfo() < 0)
                            {
                                return -1;
                            }
                        }

                        if (lastpos != DM_DENTRY_FIRST)
                        {
                            sufpos = dentry[lastpos].suffix_pos;
                        }
                        else
                        {
                            sufpos = entrance;
                        }

                        sufentry s = InitSufentry(newhash, lastpos);
                        for (int i = 0; i < hsize; i++)
                        {
                            int others;

                            if (seinfo == null || seinfo[sufpos].hashList == null)
                            {
                                throw new NullReferenceException("s.hashList is null");
                            }

                            others = seinfo[sufpos].hashList[i];
                            if (others != DM_DENTRY_NULL)
                            {
                                int tmphpos;
                                tmphpos = dentry[others].value % newhash;

                                if (s.hashList == null)
                                {
                                    throw new NullReferenceException("s.hashList is null");
                                }

                                if (s.hashList[tmphpos] == DM_DENTRY_NULL)
                                {
                                    s.hashList[tmphpos] = others;
                                }
                                else
                                {
                                    conflict = 1;
                                    break;
                                }
                            }
                        }
                        if (conflict == 0)
                        {
                            int tmphpos;
                            tmphpos = dentry[curdepos].value % newhash;

                            if (s.hashList == null)
                            {
                                throw new NullReferenceException("s.hashList is null");
                            }

                            if (s.hashList[tmphpos] == DM_DENTRY_NULL)
                            {
                                s.hashList[tmphpos] = curdepos;
                            }
                            else
                            {
                                conflict = 1;
                            }
                        }
                        if (conflict == 0)
                        {
                            if (lastpos != DM_DENTRY_FIRST)
                            {
                                dentry[lastpos].suffix_pos = seinfo.Count;
                            }
                            else
                            {
                                entrance = seinfo.Count;
                            }
                            seinfo.Add(s);
                            curpos = curdepos;
                            return 1;
                        }
                    }
                }
            }
        }

        private int ResizeInfo()
        {
            int nextentry = 0;
            int newpos = 0;
            int curde;
            int hsize;

            while (nextentry < seinfo.Count)
            {
                curde = seinfo[nextentry].backsepos;
                hsize = seinfo[nextentry].hashsize;

                if ((curde != DM_DENTRY_FIRST) && (curde >= dentry.Count))
                {
                    return -1;
                }
                if (((curde == DM_DENTRY_FIRST) && (entrance != nextentry))
                   || ((curde != DM_DENTRY_FIRST) && (dentry[curde].suffix_pos != nextentry)))
                {
                    nextentry++;
                }
                else
                {
                    if (nextentry != newpos)
                    {
                        seinfo[newpos] = seinfo[nextentry];
                        if (curde != DM_DENTRY_FIRST)
                        {
                            dentry[curde].suffix_pos = newpos;
                        }
                        else
                        {
                            entrance = newpos;
                        }
                    }
                    nextentry++;
                    newpos++;
                }
            }
            seinfo.RemoveRange(newpos, seinfo.Count - newpos);

            if (seinfo.Count > sebufsize / 2)
            {
                sebufsize *= 2;
            }
            return 1;
        }

        private int SeekLemma(string term)
        {
            int value = 0;
            int curpos = 0;
            int sufpos = 0;
            int hsize = 0;
            int hpos = 0;
            int nde = 0;

            sufpos = entrance;
            while (curpos < term.Length)
            {
                if (sufpos == DM_SUFENTRY_NULL)
                {
                    return DM_LEMMA_NULL;
                }

                value = term[curpos];
                curpos++;

                hsize = seinfo[sufpos].hashsize;
                hpos = value % hsize;

                if (seinfo == null || seinfo[sufpos].hashList == null)
                {
                    throw new NullReferenceException("seinfo is null");
                }

                nde = seinfo[sufpos].hashList[hpos];
                if ((nde == DM_DENTRY_NULL) || (dentry[nde].value != value))
                {
                    return DM_LEMMA_NULL;
                }
                sufpos = dentry[nde].suffix_pos;
            }

            return dentry[nde].lemma_pos;

        }

        private sufentry InitSufentry(int hashsize, int backsepos)
        {
            sufentry s = new sufentry();
            s.hashsize = hashsize;
            s.backsepos = backsepos;
            s.hashList = new int[hashsize];
            for (int i = 0; i < hashsize; i++)
            {
                s.hashList[i] = DM_DENTRY_NULL;
            }

            return s;
        }
    }
}