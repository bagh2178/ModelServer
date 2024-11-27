from ..models.llm import LLM
from ..tcp import TCPServer


class LLM_Server():
    def __init__(self):
        self.llm = LLM()
        self.tcpserver = TCPServer('0.0.0.0', 30083)
        self.tcpserver.set_processor(self.llm)
        self.tcpserver.start()
