import sys
import socket
import struct
import pickle
from .receive_data import receive_data


class TCPServer:
    def __init__(self, host, port, is_debugging=False):
        self.host = host
        self.port = port
        self.processor = lambda x: x
        self.is_debugging = is_debugging
        if not self.is_debugging:
            self.is_debugging = sys.gettrace() is not None

    def set_processor(self, processor):
        self.processor = processor

    def recv_and_senf(self, conn, addr):
        print(f"Connected by {addr}")
        data_length = struct.unpack('>I', conn.recv(8))[0]
        data = receive_data(conn, data_length)
        received_data = pickle.loads(data)
        print("Received data:", received_data)
        processed_data = self.processor(received_data)
        encoded_data = pickle.dumps(processed_data)
        conn.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Allow the server to bind to an address that is in TIME_WAIT state.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server listening on {self.host}:{self.port}")
            while True:
                print(f"\033[92mREADY\033[0m")
                conn, addr = s.accept()
                with conn:
                    if self.is_debugging:
                        self.recv_and_senf(conn, addr)
                    else:
                        try:
                            self.recv_and_senf(conn, addr)
                        except Exception as e:
                            print(f"An error occurred: {e}")

