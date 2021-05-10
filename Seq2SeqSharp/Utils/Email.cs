using AdvUtils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Mail;
using System.Text;
using System.Threading.Tasks;

namespace Seq2SeqSharp.Utils
{
    public class Email
    {
        public static void Send(string subject, string body, string sendFrom, string[] sendTo)
        {
            try
            {
                using (var connection = new SmtpClient("smtphost") { UseDefaultCredentials = true })
                {
                    var email = new MailMessage();
                    email.From = new MailAddress(sendFrom);
                    foreach (var toAddress in sendTo)
                    {
                        email.To.Add(toAddress);
                        email.Subject = subject;
                        email.Body = body;
                        //email.IsBodyHtml = isBodyHtml;
                        //email.Priority = mailPriority;

                        connection.Send(email);
                    }
                }
            }
            catch (Exception e)
            {
                Logger.WriteLine("Failed to send mail due to exception: " + e.ToString() + Environment.NewLine + e.Message);
            }

        }
    }
}
