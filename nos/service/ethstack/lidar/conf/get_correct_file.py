from socket import *
import binascii
import csv
import time

sock = socket(AF_INET,SOCK_STREAM)

message = bytearray()
message1 = bytearray()
SN = bytearray()
# header
message.append(0x47)
message.append(0x74)
# command
##################################################################
message.append(0x05) #0x06 for PTP-Diagnostic
##################################################################
# return code
message.append(0x00)
# Tail
message.append(0x00)
message.append(0x00)
message.append(0x00)
message.append(0x00) #0x01 payload data length for the command
message.append(0x00) #payload data value, 0x01 means PTP STATUS

sock.connect(("172.16.80.20",9347))#the ip of pandora and Port you need to call(fixed)

sock.send(message)
response=sock.recv(8)
r_cmd = int.from_bytes(response[2:3], 'big')
r_returnCode = int.from_bytes(response[3:4], 'big')
r_length = int.from_bytes(response[4:8], 'big')
response_payload=b''
while len(response_payload) < r_length:
    response_payload += sock.recv(r_length)

print(response_payload)
with open("ATD128P.dat","wb") as f:
    f.write(response_payload)
f.close()
