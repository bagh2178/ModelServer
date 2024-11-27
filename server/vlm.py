from ..models.llava import LLaVA
from ..tcp import TCPServer


class VLM_Server():
    def __init__(self):
        self.vlm = LLaVA()
        self.tcpserver = TCPServer('0.0.0.0', 40798)
        self.tcpserver.set_processor(self.vlm)
        self.tcpserver.start()
