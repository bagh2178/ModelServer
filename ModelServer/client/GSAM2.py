from ..tcp import TCPClient


class GSAM2_Client():
    def __init__(self, server_ip='127.0.0.1', server_port=7102):
        self.server_ip = server_ip
        self.server_port = server_port
        self.tcpclient = TCPClient(self.server_ip, self.server_port)

    def __call__(self, *x):
        response = self.send_and_receive(*x)
        return response
    
    def send(self, *x):
        self.tcpclient.send_data(*x)

    def send_and_receive(self, *x):
        response = self.tcpclient.send_and_receive_data(*x)
        return response
