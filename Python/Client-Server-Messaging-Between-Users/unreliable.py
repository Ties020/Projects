import socket
import re
import threading
import time


def receive():
    data = ""
    while True:
        msg = client.recv(4096).decode('utf-8')
        if not msg:
            break
        data += msg
        if "\n" in data:
            break
    if re.search('BAD-RQST-HDR', data):
        client.close()
        create_client()
    elif re.search('IN-USE', data):
        client.close()
        create_client()
    elif re.search('BAD-RQST-BODY', data):
        client.close()
        create_client()
    elif re.search('BUSY', data):
        print("Max number of clients reached")
        client.close()
        create_client()
    elif re.search('HELLO', data):
        print(f"Heyy {data[6:]}")


def send(string_bytes, num_bytes_to_send, bytes_len):
    while num_bytes_to_send > 0:
        num_bytes_to_send -= client.send(string_bytes[bytes_len - num_bytes_to_send:])
    receive()


ip = '143.47.184.219'

def xor(bit1, bit2):
    result = []
    for i in range(1, len(bit2)):
        if bit1[i] == bit2[i]:
            result.append('0')
        else:
            result.append('1')
 
    return ''.join(result)

def compute_remainder(sum, divisor):
    num_bits = len(divisor)

    to_be_divided = sum[0:num_bits]

    while num_bits < len(sum):

        if to_be_divided[0] == '1':
            to_be_divided = xor(divisor, to_be_divided) + sum[num_bits]

           
        else:
            to_be_divided = xor('00000000000000000000000000000000', to_be_divided) + sum[num_bits]
        num_bits += 1
        
    if to_be_divided[0] == '1':
            to_be_divided = xor(divisor, to_be_divided)
    else:
        to_be_divided = xor('00000000000000000000000000000000', to_be_divided)
    
    checkbits = to_be_divided
    return checkbits     

def create_client():
    global client
    ADDR = (ip, 5382)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    name = input("Please enter name: ")
    string_bytes = (f"HELLO-FROM {name}\n".encode("utf-8"))
    checkStr = string_bytes.decode("utf-8")
    

    sum = ''.join(format(ord(i), '08b') for i in checkStr)  
    divisor = "11010101010100001100110101001010"
    sum = sum + "0000000000000000000000000000000"
    
    checkbits = compute_remainder(sum, divisor)
    string_bytes = string_bytes + checkbits.encode("utf-8")
    print(string_bytes)                     #checksum appended to message
    client.sendto(string_bytes, (ip, 5382))
    receive()

create_client()


def receiveThread():
    while True:
        try:
            data = ""
            while True:
                msg = client.recv(4096).decode('utf-8')
                if not msg:
                    break
                data += msg
                if "\n" in data:
                    break
            if re.search('BAD-RQST-HDR', data):
                print("Error")
                continue
            elif re.search('SEND-OK', data):
                print("Message Sent!")
                continue
            elif re.search('DELIVERY', data):
                x = data.find(' ')
                x = x + 2
                name = data.split()
                name = name[1]
                print(f"Message received from {name}: {data[x + len(name):]}")
                continue
            elif re.search('BAD-DEST-USER', data):
                print("Bad username!")
                continue
            elif re.search('BUSY', data):
                print("Max number of clients reached")
                continue
            elif re.search("VALUE", data):
                print(data[6:])
            elif re.search("SET-OK", data):
                print(data)
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
                mess = ("!quit\n".encode("utf-8"))
                client.sendto(mess, (ip, 5382))
                client.close()
                ListenThread.join()
                break
            elif mess == '!who':
                mess = ("LIST\n".encode("utf-8"))
                client.sendto(mess, (ip, 5382))
            elif mess.startswith('@'):
                mess = mess[1:]
                mess = (f"SEND {mess} \n".encode("utf-8"))
                client.sendto(mess, (ip, 5382))
            else:
                mess = (f"{mess}\n".encode("utf-8"))
                client.sendto(mess, (ip, 5382))
        except OSError as msg:
            print(msg)
            break


ListenThread = threading.Thread(target=receiveThread)
sendThread = threading.Thread(target=sendMessThread)

ListenThread.start()
sendThread.start()