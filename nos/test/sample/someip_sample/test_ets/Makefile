CC = g++
CC1 = gcc

CFLAGS = -c -Werror -g
CFLAGS += -D SOMEIP_LINUX_PLATFORM

LDFLAGS =-lpthread -lm -ldl 
# -lasan

LIB =
INCLUDE = -I../include -I../ -I./ -I../src/ -I../src/base/netransmit/ -I../src/base/looper/ -I../src/base/ -I../src/base/object  -I../src/base/list/ -I../src/base/map/ \
	-I../src/protocol/sd/ -I../src/base/thread/ -I../src/base/transmit/ -I../src/base/file/ -I../src/base/sync/ -I../src/protocol/init/ -I../src/protocol/rpc/include/ \
	-I../src/protocol/rpc/server/ -I../src/protocol/rpc/client/ -I../src/protocol/sd/ -I../src/protocol/ipc/ -I../src/protocol/endpoint/ \
	-I../src/protocol/message/ -I../src/protocol/app_context/ -I../src/base/log/ -I../src/protocol/include/ -I../src/base/config/ -I../src/extend/network/ \
	-I../src/extend/tp/ -I../src/protocol/serializer/

all: libsomeip.a libsomeip_serializer.a someip_server someip_daemon

################################################################################################################
SRCS_SOMEIPD = ./someip_daemon.c
OBJS_SOMEIPD = $(SRCS_SOMEIPD:.c=.o)

someip_daemon:$(OBJS_SOMEIPD)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################
SRCS_SERVER = \
		./NESomeIPEtsProcess.cpp \
		./NESomeIPTc8Process.cpp \
		./NESomeIPTc8Server.cpp \

OBJS_SERVER = $(SRCS_SERVER:.cpp=.o)

someip_server:$(OBJS_SERVER)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a libsomeip_serializer.a -lpthread

%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< -std=c++11

###############################################################################

CFLAGS3 = -c -Wall -g -fPIC -Wimplicit-function-declaration -Werror=return-type -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast \
		-Werror=address -Werror=sequence-point -Werror=delete-non-virtual-dtor -Werror=array-bounds \
		-Werror=sizeof-pointer-memaccess -Werror=format-security \
		-Werror=implicit-function-declaration -Werror=int-conversion \
		-Werror=date-time 
# 		-fsanitize=address -fno-omit-frame-pointer
CFLAGS3 += -D SOMEIP_LINUX_PLATFORM
CFLAGS3 += -D SOMEIP_PC_PLATFORM
LDFLAGS3 =-lpthread -lm -ldl -lcrypto -static 
# -static-lasan

SRCS_SOMEIP = \
        ../src/base/sync/ne_someip_sync_obj.c \
		../src/base/sync/ne_someip_sync_wait_obj.c \
		../src/base/map/ne_someip_map.c \
		../src/base/list/ne_someip_list.c \
		../src/base/looper/ne_someip_looper.c \
		../src/base/looper/ne_someip_thread.c \
		../src/base/log/ne_someip_log.c \
		../src/base/config/cJSON.c \
		../src/base/config/ne_someip_config_parse.c \
		../src/base/file/ne_someip_file.c \
		../src/base/transmit/ne_someip_transmit.c \
		../src/base/transmit/ne_someip_transmitcore.c \
		../src/base/transmit/ne_someip_transmitimpltcp.c \
		../src/base/transmit/ne_someip_transmitimpludp.c \
		../src/base/transmit/ne_someip_transmitimplunixDomain.c \
		../src/protocol/endpoint/ne_someip_endpoint_common_func.c \
		../src/protocol/endpoint/ne_someip_endpoint_unix.c \
		../src/protocol/endpoint/ne_someip_endpoint_tcp_data.c \
		../src/protocol/endpoint/ne_someip_endpoint_udp_data.c \
		../src/protocol/endpoint/ne_someip_endpoint_udp_sd.c \
		../src/protocol/endpoint/ne_someip_endpoint_core.c \
		../src/protocol/endpoint/ne_someip_endpoint_runnable_func.c \
		../src/protocol/endpoint/ne_someip_endpoint_tool.c \
		../src/protocol/endpoint/ne_someip_endpoint_recv.c \
		../src/protocol/message/ne_someip_message.c \
		../src/protocol/message/ne_someip_serializer.c \
		../src/protocol/message/ne_someip_deserializer.c \
		../src/protocol/message/ne_someip_sd_message.c \
		../src/protocol/rpc/server/ne_someip_server_context.c \
		../src/protocol/rpc/server/ne_someip_server_tool.c \
		../src/protocol/rpc/server/ne_someip_provided_service_instance.c \
		../src/protocol/rpc/server/ne_someip_provided_event_behaviour.c \
		../src/protocol/rpc/server/ne_someip_provided_eventgroup_behaviour.c \
		../src/protocol/rpc/server/ne_someip_provided_method_behaviour.c \
		../src/protocol/rpc/client/ne_someip_client_tool.c \
		../src/protocol/rpc/server/ne_someip_server_context.c \
		../src/protocol/rpc/server/ne_someip_server_tool.c \
		../src/protocol/init/ne_someip_daemon.c \
		../src/protocol/ipc/ne_someip_ipc_behaviour.c \
		../src/protocol/ipc/ne_someip_ipc_daemon_behaviour.c \
		../src/protocol/sd/ne_someip_sd.c \
		../src/protocol/sd/ne_someip_sd_tool.c \
		../src/protocol/sd/ne_someip_sd_msg.c \
		../src/protocol/app_context/ne_someip_app_context.c \
		../src/protocol/rpc/client/ne_someip_client_context.c \
		../src/protocol/rpc/client/ne_someip_common_service_instance.c \
		../src/protocol/rpc/client/ne_someip_required_service_instance.c \
		../src/protocol/rpc/client/ne_someip_required_event_behaviour.c \
		../src/protocol/rpc/client/ne_someip_required_eventgroup_behaviour.c \
		../src/protocol/rpc/client/ne_someip_required_find_service_behaviour.c \
		../src/protocol/rpc/client/ne_someip_required_method_behaviour.c \
		../src/protocol/rpc/client/ne_someip_required_network_connect_behaviour.c \
		../src/protocol/rpc/client/ne_someip_required_service_connect_behaviour.c \
		../src/protocol/rpc/client/ne_someip_client_runnable_func.c \
		../src/extend/network/ne_someip_network_monitor.c \
		../src/extend/tp/ne_someip_tp.c \


OBJS_SOMEIP = $(SRCS_SOMEIP:.c=.o)

libsomeip.a:$(OBJS_SOMEIP)
	ar cr $@ $^

%.o:%.c
	$(CC1) $(CFLAGS3) $(INCLUDE) -o $@ -c $<
###############################################################################
CFLAGS1 = -c -Wall -g
CFLAGS1 += -D SOMEIP_LINUX_PLATFORM
CFLAGS1 += -D SOMEIP_PC_PLATFORM

SRCS = \
	../src/protocol/serializer/NESomeIPPayloadDeserializeGeneral.cpp \
	../src/protocol/serializer/NESomeIPPayloadSerializeGeneral.cpp \
	../src/protocol/serializer/NESomeIPPayloadUtils.cpp \

OBJS = $(SRCS:.cpp=.o)

libsomeip_serializer.a:$(OBJS)
	ar cr $@ $^

%.o:%.cpp
	$(CC) $(CFLAGS1) $(INCLUDE) -o $@ -c $< -std=c++14
###############################################################################

clean:
	rm -rf libsomeip.a $(OBJS_SOMEIP) libsomeip_serializer.a $(OBJS) *.o someip_server $(OBJS_SERVER) someip_daemon $(OBJS_SOMEIPD)