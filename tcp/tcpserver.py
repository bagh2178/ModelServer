import socket
import struct
import pickle
from .receive_data import receive_data


class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.processor = lambda x: x

    def set_processor(self, processor):
        self.processor = processor

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
                    # try:
                        print(f"Connected by {addr}")
                        data_length = struct.unpack('>I', conn.recv(4))[0]
                        data = receive_data(conn, data_length)
                        received_data = pickle.loads(data)
                        print("Received data:", received_data)
                        processed_data = self.processor(received_data)
                        encoded_data = pickle.dumps(processed_data)
                        conn.sendall(struct.pack('>I', len(encoded_data)) + encoded_data)
                    # except Exception as e:
                        # print(f"An error occurred: {e}")

