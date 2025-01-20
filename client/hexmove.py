from ..tcp import TCPClient


class Hexmove_Client():
    def __init__(self):
        self.tcpclient = TCPClient('127.0.0.1', 7002)

    def __call__(self, *x):
        response = self.send_and_receive(*x)
        return response
    
    def send(self, *x):
        self.tcpclient.send_data(*x)

    def send_and_receive(self, *x):
        response = self.tcpclient.send_and_receive_data(*x)
        return response
