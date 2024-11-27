from .tcp import TCPClient


class Agent_Client():
    def __init__(self):
        self.tcpclient = TCPClient('127.0.0.1', 7002)

    def __call__(self, prompt):
        response = self.tcpclient.send_and_receive_data(prompt)
        response = response[0]
        return response
