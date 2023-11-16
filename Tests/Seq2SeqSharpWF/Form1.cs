using AdvUtils;
using Seq2SeqSharp.Applications;
using Seq2SeqSharp.Corpus;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace Seq2SeqSharpWF
{
    public partial class Form1 : Form
    {
        private static Form1 mainForm;

        public Form1()
        {
            mainForm = this;
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Logger.Initialize(Logger.Destination.Callback | Logger.Destination.Logfile, Logger.Level.err | Logger.Level.warn | Logger.Level.info | Logger.Level.debug, "Seq2SeqSharpWF_Train.log", callback);

            Logger.WriteLine(Logger.Level.info, "-----------------------------------------------------------------------------------------------------------------------------------");
            Logger.WriteLine(Logger.Level.info, "This application demonstrates how to use the callback meachanism in Seq2SeqSharp in a WinForms app.");
            Logger.WriteLine(Logger.Level.info, "-----------------------------------------------------------------------------------------------------------------------------------");

            try
            {
                Seq2SeqOptions opts = new Seq2SeqOptions();
                DecodingOptions decodingOptions = opts.CreateDecodingOptions();
                Seq2Seq ss = null;

                // simulating progress from Seq2SeqSharp - if you setup the parameters well below, then this is of course not needed
                for (int i = 1; i < 100; i++)
                {
                    Logger.WriteLine(Logger.Level.info, "", i);
                }

                // Load train corpus
                Seq2SeqCorpus trainCorpus = new Seq2SeqCorpus(corpusFilePath: opts.TrainCorpusPath,
                                                    srcLangName: opts.SrcLang,
                                                    tgtLangName: opts.TgtLang,
                                                    maxTokenSizePerBatch: opts.MaxTokenSizePerBatch,
                                                    maxSrcSentLength: opts.MaxSrcSentLength,
                                                    maxTgtSentLength: opts.MaxTgtSentLength,
                                                    shuffleEnums: opts.ShuffleType,
                                                    tooLongSequence: opts.TooLongSequence);

                // Note: this call will fail because the parameters are not setup well. The aim of this application is to show you how to use the callback mechanism in Seq2SeqSharp
            }
            catch (Exception ex)
            {
                Logger.WriteLine(Logger.Level.err, ex.Message);
                return;
            }

            Logger.WriteLine(Logger.Level.info, "Ready!");
        }

        private static void SetProgressValue(int value)
        {
            if (mainForm.InvokeRequired)
            {
                mainForm.Invoke((MethodInvoker)delegate
                {
                    SetProgressValue(value);
                });
            }
            else
            {
                mainForm.progressBar1.Value = value;
            }
        }

        private static void SetLog(string text)
        {
            if (mainForm.InvokeRequired)
            {
                mainForm.Invoke((MethodInvoker)delegate
                {
                    SetLog(text);
                });
            }
            else
            {
                mainForm.textBox1.AppendText(text);
            }
        }

        public ProgressCallback callback =
            (value, log, mtype, color) =>
            {
                // value > 0 -> progress reporting | 0 -> no progress reporting, but log message
                if (value > 0)
                {
                    SetProgressValue(value);
                }
                else
                {
                    if (log.ToString() == "") return 0;

                    switch ((Logger.Level)mtype)
                    {
                        case Logger.Level.info:
                            SetLog("INFO: " + log.ToString() + Environment.NewLine);
                            break;
                        case Logger.Level.err:
                            SetLog("ERROR: " + log.ToString() + Environment.NewLine);
                            break;
                        case Logger.Level.warn:
                            SetLog("WARNING: " + log.ToString() + Environment.NewLine);
                            break;
                        case Logger.Level.debug:
#if DEBUG
                            SetLog("DEBUG: " + log.ToString() + Environment.NewLine);
#endif
                            break;
                    }
                }

                return 0;
            };
    }


}

