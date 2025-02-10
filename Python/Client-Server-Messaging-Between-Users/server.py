import socket
import threading
import time

ip = "0.0.0.0"
port = 5500

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind((ip,port))

print("Server starting...")

tupleList = []
name_List = []
addresses = []
special_characters = ["\"", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "+", "?", "_", "=", ",", "<", ">", "/"]

def new_client(conn, addr):
    connected = True
    checkName = True
    wrongName = False
    print("1")
    try:
        while connected:
            msg = conn.recv(4096).decode('utf-8')
            print("2")


            if not msg:
                print("3")
                pass

            if checkName:
                msg = msg[11:]
                msg = msg[:-1]

                for s in msg:
                    for n in special_characters:
                        if s == n:
                            mess = ("BAD-RQST-HDR\n".encode('utf-8'))
                            bytes_len = len(mess)
                            num_bytes_to_send = bytes_len
                            while num_bytes_to_send > 0:
                                num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                                connected = False
                                wrongName = True
                                conn.close()
                                break
                        if wrongName:
                            break
                    if wrongName:
                        break

                for i in name_List:
                    if msg == i:
                        conn.send("IN-USE\n".encode('utf-8'))
                        connected = False
                        conn.close()
                        break

                checkName = False
                if wrongName == False:
                    portUser = addr[1]
                    tupleUser = (msg, portUser)
                    tupleList.append(tupleUser)
                    name_List.append(msg)
                    addresses.append(conn)

                mess = (f"HELLO {msg}".encode('utf-8'))
                bytes_len = len(mess)
                num_bytes_to_send = bytes_len
                while num_bytes_to_send > 0:
                    num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                continue


            if checkName == False:
                if msg == "!quit":
                    conn.close()
                    connected = False
                    break
                elif msg == "LIST\n":
                    names = ','.join(map(str, name_List))
                    mess = (f"LIST-OK {names}".encode('utf-8'))
                    bytes_len = len(mess)
                    num_bytes_to_send = bytes_len
                    while num_bytes_to_send > 0:
                        num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                elif len(name_List) > 64:
                    mess = ("BUSY\n".encode('utf-8'))
                    bytes_len = len(mess)
                    num_bytes_to_send = bytes_len
                    while num_bytes_to_send > 0:
                        num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                elif msg[:4] == "SEND":
                    goodName = False
                    dest_user = msg.split()
                    dest_user = dest_user[1]
                    index = 0
                    for names in name_List:
                        if names == dest_user:
                            mess = ("SEND-OK\n".encode('utf-8'))
                            bytes_len = len(mess)
                            num_bytes_to_send = bytes_len
                            while num_bytes_to_send > 0:
                                num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                            time.sleep(0.01)
                            target = addresses[index]
                            mess = (f"DELIVERY {msg[5:]}\n".encode('utf-8'))
                            bytes_len = len(mess)
                            num_bytes_to_send = bytes_len
                            while num_bytes_to_send > 0:
                                num_bytes_to_send -= target.send(mess[bytes_len - num_bytes_to_send:])                          
                            goodName = True
                        index = index + 1
                    if goodName == False:
                        mess = ("BAD-DEST-USER\n".encode('utf-8'))
                        bytes_len = len(mess)
                        num_bytes_to_send = bytes_len
                        while num_bytes_to_send > 0:
                            num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])
                else:
                    mess = ("BAD-RQST-HDR\n".encode('utf-8'))
                    bytes_len = len(mess)
                    num_bytes_to_send = bytes_len
                    while num_bytes_to_send > 0:
                        num_bytes_to_send -= conn.send(mess[bytes_len - num_bytes_to_send:])


            print(f"[{addr}] {msg}")

    except:
        index = 0
        port_user = addr[1]

        for t in tupleList:
            t = t[1]
            if port_user == t:
                name = name_List[index]
                name_List.pop(index)
                tupleList.pop(index)
                print(f"User {name} left!")
            index = index + 1
        conn.close()

def start():
    server.listen(64)
    print("started")
    while True:
        print("started")

        conn, addr = server.accept()

        print("1")

        print("Connection from: " + str(addr))
        new_thread = threading.Thread(target=new_client, args=(conn, addr))
        new_thread.start()

start()