from flask_mail import Message

def send_email(mail, subject, recipients, body):
    """Send email using Flask-Mail"""
    try:
        if not recipients:
            print("⚠️ No recipients provided for email.")
            return False

        # Ensure recipients is always a list
        if isinstance(recipients, str):
            recipients = [recipients]

        msg = Message(subject, recipients=recipients)
        msg.body = body
        mail.send(msg)
        print(f"✅ Email sent to {recipients}")
        return True

    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False
