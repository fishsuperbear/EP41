import socket
import ctypes

# Define the LogTime structure
class LogTime(ctypes.Structure):
    _fields_ = [
        ("sec", ctypes.c_uint32),
        ("nsec", ctypes.c_uint32)
    ]

# Define the McuLogHeader structure
class McuLogHeader(ctypes.Structure):
    _fields_ = [
        ("app_id", ctypes.c_uint8),
        ("ctx_id", ctypes.c_uint8),
        ("level", ctypes.c_uint8),
        ("seq", ctypes.c_uint8),
        ("stamp", LogTime),  # Use LogTime structure here
        ("length", ctypes.c_uint32)
    ]

# Define the McuLog structure
class McuLog(ctypes.Structure):
    _fields_ = [
        ("header", McuLogHeader),
        ("log", ctypes.c_uint8 * 255)
    ]

# Create an McuLog structure instance and set default values
mcu_log = McuLog()
mcu_log.header.app_id = 0x03
mcu_log.header.ctx_id = 0x00
mcu_log.header.level = 0x04
mcu_log.header.seq = 0x09
mcu_log.header.stamp.sec = 0x13
mcu_log.header.stamp.nsec = 0x14
log_data = b"Create an McuLo structure instance stture instance and sgggeo sto struo structure instance and secture instance and seruo struco structure instance and seture instance and secture instance and seo structure instano structure instance and sece and se"
mcu_log.header.length = len(log_data)
mcu_log.log[:len(log_data)] = log_data

# Target host and port
target_host = "10.6.72.161"
target_port = 23456

# Serialize the structure data
serialized_data = bytearray(mcu_log)

# Create a UDP socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send the data
udp_socket.sendto(serialized_data, (target_host, target_port))

# Close the socket
udp_socket.close()


