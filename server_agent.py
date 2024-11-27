from .models.agent import Agent
from .tcp import TCPServer


class Agent_Server():
    def __init__(self):
        self.agent = Agent()
        self.tcpserver = TCPServer('0.0.0.0', 7002)
        self.tcpserver.set_processor(self.agent)
        self.tcpserver.start()
