from ..models.hexmove.hexmove import Hexmove
from ..tcp import TCPServer


class Hexmove_Server():
    def __init__(self):
        self.hexmove = Hexmove()
        self.tcpserver = TCPServer('0.0.0.0', 7002)
        self.tcpserver.set_processor(self.hexmove)
        self.tcpserver.start()
