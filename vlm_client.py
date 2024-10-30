from .tcp import TCPClient


class VLM_Client():
    def __init__(self):
        self.tcpclient = TCPClient('127.0.0.1', 40798)

    def __call__(self, query, image):
        response = self.tcpclient.send_and_receive_data(query, image)
        response = response[0]
        return response
