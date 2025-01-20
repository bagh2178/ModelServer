import socket
import struct
import pickle
from .receive_data import receive_data


class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_data(self, *data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            encoded_data = pickle.dumps(data)
            s.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)

    def send_and_receive_data(self, *data):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            encoded_data = pickle.dumps(data)
            s.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)
            data_length = struct.unpack('>I', s.recv(4))[0]
            response_data = receive_data(s, data_length)
            return pickle.loads(response_data)

