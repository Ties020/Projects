import socket
import re
import threading
import time

ip = "0.0.0.0"

def receive():
    data = client.recv(4096).decode('utf-8')
    if re.search('BAD-RQST-HDR', data):
        client.close()
        create_client()
    elif re.search('IN-USE', data):
        client.close()
        create_client()
    elif re.search('BAD-RQST-BODY', data):
        client.close()
        create_client()
    elif re.search('HELLO', data):
        print(f"Heyy {data[6:]}")


def send(string_bytes, num_bytes_to_send, bytes_len):
    while num_bytes_to_send > 0:
        num_bytes_to_send -= client.send(string_bytes[bytes_len - num_bytes_to_send:])
    receive()


def create_client():
    print("starting")
    global client
    ADDR = (ip, 5500)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client.connect(ADDR)
    print("starting")

    name = input("Please enter name: ")
    name = name.split()
    string_bytes = (f"HELLO-FROM {name[0]}\n".encode("utf-8"))
    bytes_len = len(string_bytes)
    num_bytes_to_send = bytes_len
    send(string_bytes, num_bytes_to_send, bytes_len)


create_client()

def receiveThread():
    while True:
        try:
            data = client.recv(4096).decode('utf-8')
            if not data:
                break
            else:
                if re.search('BAD-RQST-HDR', data):
                    print("Error")
                    continue
                elif re.search('SEND-OK', data):
                    print("Message Sent!")
                    continue
                elif re.search('DELIVERY', data):
                    x = data.find(' ')
                    x = x+2
                    name = data.split()
                    name = name[1]
                    print(f"Message received from {name}: {data[x+len(name):]}")
                    continue
                elif re.search('BAD-DEST-USER', data):
                    print("Bad username!")
                    continue
                elif re.search('BUSY', data):
                    print("Max number of clients reached")
                    continue
                elif re.search('LIST-OK', data):
                    print(data[8:])
                    continue

        except:
            break

def sendMessThread():
    while True:
        time.sleep(0.5)
        try:
            mess = input("Enter message: ")
            if mess == '!quit':
                client.close()
                ListenThread.join()
                break
            elif mess == '!who':
                mess = ("LIST\n".encode("utf-8"))
                bytes_len = len(mess)
                num_bytes_to_send = bytes_len
                while num_bytes_to_send > 0:
                    num_bytes_to_send -= client.send(mess[bytes_len - num_bytes_to_send:])
            elif mess.startswith('@'):
                mess = mess[1:]
                mess = (f"SEND {mess} \n".encode("utf-8"))
                bytes_len = len(mess)
                num_bytes_to_send = bytes_len
                while num_bytes_to_send > 0:
                    num_bytes_to_send -= client.send(mess[bytes_len - num_bytes_to_send:])
            else:
                mess = (f"{mess}\n".encode("utf-8"))
                bytes_len = len(mess)
                num_bytes_to_send = bytes_len
                while num_bytes_to_send > 0:
                    num_bytes_to_send -= client.send(mess[bytes_len - num_bytes_to_send:])
        except OSError as msg:
            print(msg)
            break


ListenThread = threading.Thread(target=receiveThread)
sendThread = threading.Thread(target=sendMessThread)

ListenThread.start()
sendThread.start()