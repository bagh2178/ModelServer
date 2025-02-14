from ..models.hexmove.hexmove import Hexmove
from ..tcp import TCPServer


class Hexmove_Server():
    def __init__(self, is_debugging=False):
        self.hexmove = Hexmove()
        self.tcpserver = TCPServer('0.0.0.0', 7002, is_debugging=is_debugging)
        self.tcpserver.set_processor(self.hexmove)
        self.tcpserver.start()
