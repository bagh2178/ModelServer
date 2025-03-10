from ..models.GSAM2.GSAM2 import GSAM2
from ..tcp import TCPServer


class GSAM2_Server():
    def __init__(self, server_ip='127.0.0.1', server_port=7003, is_debugging=False):
        self.GSAM2 = GSAM2()
        self.tcpserver = TCPServer(server_ip, server_port, is_debugging=is_debugging)
        self.tcpserver.set_processor(self.GSAM2)
        self.tcpserver.start()
