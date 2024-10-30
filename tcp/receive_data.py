def receive_data(sock, expected_length):
    """
    Receive exactly the specified amount of data from the socket.
    :param sock: The socket object.
    :param expected_length: The number of bytes to receive.
    :return: The received data as bytes.
    """
    received_data = bytearray()
    while len(received_data) < expected_length:
        chunk = sock.recv(expected_length - len(received_data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        received_data.extend(chunk)
    return bytes(received_data)