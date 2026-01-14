from twilio.rest import Client

class SmsSender:
    """
    Simple Twilio SMS wrapper.
    """
    def __init__(self,
                 account_sid: str,
                 auth_token:  str,
                 from_number: str):
        self.client      = Client(account_sid, auth_token)
        self.from_number = from_number

    def send_sms(self, to_number: str, body: str) -> str:
        """
        Sends `body` to `to_number`. Returns the Twilio message SID.
        """
        print(to_number, body)
        msg = self.client.messages.create(
            body=body,
            from_=self.from_number,
            to=to_number
        )
        return msg.sid
