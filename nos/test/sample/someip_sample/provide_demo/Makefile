# /**
# 
# * Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.
# 
# * All Rights Reserved.
# 
# *
# 
# * Redistribution and use in source and binary forms, with or without
# 
# * modification, are NOT permitted except as agreed by
# 
# * iAuto Software(Shanghai) Co., Ltd.
# 
# *
# 
# * Unless required by applicable law or agreed to in writing, software
# 
# * distributed under the License is distributed on an "AS IS" BASIS,
# 
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 
# */
CC = gcc

CFLAGS = -c -Werror -g
CFLAGS += -D SOMEIP_LINUX_PLATFORM

LDFLAGS =-lpthread -lm -ldl -L./openssl-1.1.1i

LIB =
INCLUDE = -I../include -I../ -I./ -I../src/ -I../src/base/netransmit/ -I../src/base/looper/ -I../src/base/ -I../src/base/object  -I../src/base/list/ -I../src/base/map/ \
	-I../src/protocol/sd/ -I../src/base/thread/ -I../src/base/transmit/ -I../src/base/file/ -I../src/base/sync/ -I../src/protocol/init/ -I../src/protocol/rpc/include/ \
	-I../src/protocol/rpc/server/ -I../src/protocol/rpc/client/ -I../src/protocol/rpc/common_reuse/ -I../src/protocol/sd/ -I../src/protocol/ipc/ -I../src/protocol/endpoint/ \
	-I../src/protocol/message/ -I../src/protocol/app_context/ -I../src/base/log/ -I../src/protocol/include/ -I../src/base/config/ -I../src/extend/network/ \
	-I../src/extend/tp/  -I./openssl-1.1.1i/include -I../src/base/pt_trace

all: libsomeip.a tst_someip_ipc tst_someip_daemon tst_someip_endpoint_client tst_someip_endpoint_server tst_someip_req_instance \
	tst_someip_provided_instance tst_someip_config tst_network tst_udp tst_someip_cmdline_pro tst_someip_cmdline_req \
	tst_someip_cmdline_pro1 tst_someip_cmdline_req1 tst_someip_pro_usage tst_someip_req_usage

SRCS_udp = ./udp_test.c

OBJS_udp = $(SRCS_udp:.c=.o)

tst_udp:$(OBJS_udp)
	$(CC) -o $@ $^

%.o:%.c
	$(CC) $(CFLAGS)  $(INCLUDE) -o $@ -c $<

###############################################################################################

SRCS1 = ./someip_server.c

OBJS1 = $(SRCS1:.c=.o)

tst_someip_server:$(OBJS1)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread

%.o:%.c
	$(CC) $(CFLAGS)  $(INCLUDE) -o $@ -c $<

###############################################################################################

SRCS2 = ./someip_client.c

OBJS2 = $(SRCS2:.c=.o)

tst_someip_client:$(OBJS2)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS5 = ./someip_daemon.c

OBJS5 = $(SRCS5:.c=.o)

tst_someip_daemon:$(OBJS5)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_NEOBJECT = ./someip_object.c

OBJS_NEOBJECT = $(SRCS_NEOBJECT:.c=.o)

tst_someip_object:${OBJS_NEOBJECT}
	$(CC) -o $@ $^ $(LDFLAGS) -lpthread

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<
################################################################################################################

SRCS_NELIST = ./someip_list.c ../src/base/list/ne_someip_list.c

OBJS_NELIST = $(SRCS_NELIST:.c=.o)

tst_someip_list:${OBJS_NELIST}
	$(CC) -o $@ $^

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_NEMAP = ./someip_map.c ../src/base/map/ne_someip_map.c

OBJS_NEMAP = $(SRCS_NEMAP:.c=.o)

tst_someip_map:${OBJS_NEMAP}
	$(CC) -o $@ $^

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_NETHREAD = ./someip_thread.c ../src/base/looper/ne_someip_thread.c ../src/base/looper/ne_someip_looper.c ../src/base/list/ne_someip_list.c \
	../src/base/log/ne_someip_log.c

OBJS_NETHREAD = $(SRCS_NETHREAD:.c=.o)

tst_someip_thread:${OBJS_NETHREAD}
	$(CC) -o $@ $^ $(LDFLAGS)

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<  $(LDFLAGS)

################################################################################################################

SRCS_NELOOPER = \
	./someip_looper.c \
	../src/base/looper/ne_someip_thread.c \
	../src/base/looper/ne_someip_looper.c \
	../src/base/list/ne_someip_list.c \
	../src/base/log/ne_someip_log.c

OBJS_NELOOPER = $(SRCS_NELOOPER:.c=.o)

tst_someip_looper:${OBJS_NELOOPER}
	$(CC) -o $@ $^ $(LDFLAGS)

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS)

################################################################################################################

SRCS_SD_ipc = ./someip_ipc.c

OBJS_SD_ipc = $(SRCS_SD_ipc:.c=.o)

tst_someip_ipc:$(OBJS_SD_ipc)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_P_ins = ./someip_provided_instance.c

OBJS_P_ins = $(SRCS_P_ins:.c=.o)

tst_someip_provided_instance:$(OBJS_P_ins)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_network = ./network_test.c

OBJS_network = $(SRCS_network:.c=.o)

tst_network:$(OBJS_network)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_NETRANSMIT = \
		../src/base/map/ne_someip_map.c \
		../src/base/list/ne_someip_list.c \
		../src/base/looper/ne_someip_thread.c \
		../src/base/looper/ne_someip_looper.c \
		../src/base/log/ne_someip_log.c \
		../src/base/sync/ne_someip_sync_obj.c \
		../src/base/transmit/ne_someip_transmit.c \
		../src/base/transmit/ne_someip_transmitcore.c \
		../src/base/transmit/ne_someip_transmitimpltcp.c \
		../src/base/transmit/ne_someip_transmitimpludp.c \
		../src/base/transmit/ne_someip_transmitimpltls.c \
		../src/base/transmit/ne_someip_transmitimpldtls.c \
		../src/base/transmit/ne_someip_transmitimplunixDomain.c \
		../src/base/file/ne_someip_file.c
################################################################################################################
SRCS_NETRANSMIT_TCP_SERVER = \
		./someip_transmit_tcp_server.c \
		$(SRCS_NETRANSMIT)

OBJS_NETRANSMIT_TCP_SERVER = $(SRCS_NETRANSMIT_TCP_SERVER:.c=.o)

tst_someip_transmit_tcp_server:${OBJS_NETRANSMIT_TCP_SERVER}
	$(CC) -o $@ $^ $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m  -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

################################################################################################################

SRCS_NETRANSMIT_TCP_CLIENT = \
		./someip_transmit_tcp_client.c \
		$(SRCS_NETRANSMIT)

OBJS_NETRANSMIT_TCP_CLIENT = $(SRCS_NETRANSMIT_TCP_CLIENT:.c=.o)

tst_someip_transmit_tcp_client:${OBJS_NETRANSMIT_TCP_CLIENT}
	$(CC) -o $@ $^ $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

################################################################################################################

SRCS_NETRANSMIT_UDP = \
		./someip_transmit_udp.c \
		$(SRCS_NETRANSMIT)

OBJS_NETRANSMIT_UDP = $(SRCS_NETRANSMIT_UDP:.c=.o)

tst_someip_transmit_udp:${OBJS_NETRANSMIT_UDP}
	$(CC) -o $@ $^ $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

################################################################################################################

SRCS_NETRANSMIT_UNIX_CLIENT = \
		./someip_transmit_unix_client.c \
		$(SRCS_NETRANSMIT)

OBJS_NETRANSMIT_UNIX_CLIENT = $(SRCS_NETRANSMIT_UNIX_CLIENT:.c=.o)

tst_someip_transmit_unix_client:${OBJS_NETRANSMIT_UNIX_CLIENT}
	$(CC) -o $@ $^ $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

################################################################################################################

SRCS_NETRANSMIT_UNIX_SERVER = \
		./someip_transmit_unix_server.c \
		$(SRCS_NETRANSMIT)

OBJS_NETRANSMIT_UNIX_SERVER = $(SRCS_NETRANSMIT_UNIX_SERVER:.c=.o)

tst_someip_transmit_unix_server:${OBJS_NETRANSMIT_UNIX_SERVER}
	$(CC) -o $@ $^ $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< $(LDFLAGS) -L/usr/lib -I/usr/lib/python3.5 -lpython3.5m

################################################################################################################

SRCS_ENDPOINT_CLIENT = ./someip_endpoint_client.c

OBJS_ENDPOINT_CLIENT = $(SRCS_ENDPOINT_CLIENT:.c=.o)

tst_someip_endpoint_client:$(OBJS_ENDPOINT_CLIENT)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS)  $(INCLUDE) -o $@ -c $< 

################################################################################################################

SRCS_ENDPOINT_SERVER = ./someip_endpoint_server.c

OBJS_ENDPOINT_SERVER = $(SRCS_ENDPOINT_SERVER:.c=.o)

tst_someip_endpoint_server:$(OBJS_ENDPOINT_SERVER)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

SRCS_REQ_INSTANCE = ./someip_req_instance.c 

OBJS_REQ_INSTANCE = $(SRCS_REQ_INSTANCE:.c=.o)

tst_someip_req_instance:$(OBJS_REQ_INSTANCE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

###############################################################################################################

SRCS_PRO_COMMANDLINE = ./someip_test_by_cmdline_pro.c

OBJS_PRO_COMMANDLINE = $(SRCS_PRO_COMMANDLINE:.c=.o)

tst_someip_cmdline_pro:$(OBJS_PRO_COMMANDLINE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<


###############################################################################################################

SRCS_REQ_COMMANDLINE = ./someip_test_by_cmdline_req.c

OBJS_REQ_COMMANDLINE = $(SRCS_REQ_COMMANDLINE:.c=.o)

tst_someip_cmdline_req:$(OBJS_REQ_COMMANDLINE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

###############################################################################################################

SRCS_REQ1_COMMANDLINE = ./someip_req_instance1.c

OBJS_REQ1_COMMANDLINE = $(SRCS_REQ1_COMMANDLINE:.c=.o)

tst_someip_cmdline_req1:$(OBJS_REQ1_COMMANDLINE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<


###############################################################################################################

SRCS_PRO1_COMMANDLINE = ./someip_provided_instance1.c

OBJS_PRO1_COMMANDLINE = $(SRCS_PRO1_COMMANDLINE:.c=.o)

tst_someip_cmdline_pro1:$(OBJS_PRO1_COMMANDLINE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

###############################################################################################################

SRCS_PRO_USAGE = ./someip_provided_usage_demo.c

OBJS_PRO_USAGE = $(SRCS_PRO_USAGE:.c=.o)

tst_someip_pro_usage:$(OBJS_PRO_USAGE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

###############################################################################################################

SRCS_REQ_USAGE = ./someip_required_usage_demo.c

OBJS_REQ_USAGE = $(SRCS_REQ_USAGE:.c=.o)

tst_someip_req_usage:$(OBJS_REQ_USAGE)
	$(CC) -o $@ $^ $(LDFLAGS) libsomeip.a -lpthread -zmuldefs -lcrypto -lssl

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

###############################################################################################################

SRCS_NECONFIG = \
	../src/base/log/ne_someip_log.c \
	../src/base/config/cJSON.c \
	../src/base/config/ne_someip_config_parse.c \
	./someip_config.c

OBJS_NECONFIG = $(SRCS_NECONFIG:.c=.o)

tst_someip_config:${OBJS_NECONFIG}
	$(CC) -o $@ $^ $(LDFLAGS)

%.o:%.c
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $<

################################################################################################################

CFLAGS3 = -c -Wall -g -fPIC -Wimplicit-function-declaration -Werror=return-type -Werror=int-to-pointer-cast -Werror=pointer-to-int-cast \
		-Werror=address -Werror=sequence-point -Werror=delete-non-virtual-dtor -Werror=array-bounds \
		-Werror=sizeof-pointer-memaccess -Werror=format-security \
		-Werror=implicit-function-declaration -Werror=int-conversion \
		-Werror=date-time
CFLAGS3 += -D SOMEIP_LINUX_PLATFORM
CFLAGS3 += -D SOMEIP_PC_PLATFORM
CFLAGS3 += -D__SOMEIP_FILE_NAME__=\"$(subst $(dir $<),,$<)\"
LDFLAGS3 =-lpthread -lm -ldl -lcrypto -static

SRCS3 = \
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
		../src/base/transmit/ne_someip_transmitimpltls.c \
		../src/base/transmit/ne_someip_transmitimpldtls.c \
		../src/base/transmit/ne_someip_transmitimplunixDomain.c \
		../src/extend/tp/ne_someip_tp.c \
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
		../src/protocol/rpc/common_reuse/ne_someip_common_reuse_manager.c \
		../src/protocol/rpc/common_reuse/ne_someip_common_reuse_tool.c \
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


OBJS3 = $(SRCS3:.c=.o)

libsomeip.a:$(OBJS3)
	ar cr $@ $^

%.o:%.c
	$(CC) $(CFLAGS3) $(INCLUDE) -o $@ -c $<
###############################################################################

clean:
	rm -rf libsomeip.a $(OBJS3) tst_someip_server $(OBJS1) tst_someip_client $(OBJS2) *.o tst_someip_daemon tst_someip_object $(OBJS_NEOBJECT) \
	tst_someip_list $(OBJS_NELIST) tst_someip_map $(OBJS_NEMAP) tst_someip_looper $(OBJS_NELOOPER) tst_someip_thread $(OBJS_NETHREAD) \
	tst_someip_transmit ${OBJS_NETRANSMIT_TCP_SERVER} tst_someip_transmit_tcp_server ${OBJS_NETRANSMIT_TCP_CLIENT} tst_someip_transmit_tcp_client \
	tst_someip_endpoint_client $(OBJS_ENDPOINT_CLIENT) tst_someip_endpoint_server $(OBJS_ENDPOINT_SERVER) tst_someip_transmit_udp ${OBJS_NETRANSMIT_UDP} \
	tst_someip_transmit_unix_client ${OBJS_NETRANSMIT_UNIX_CLIENT} tst_someip_transmit_unix_server ${OBJS_NETRANSMIT_UNIX_SERVER} \
	$(OBJS_SD_ipc) tst_someip_ipc $(OBJS_REQ_INSTANCE) tst_someip_req_instance \
	$(OBJS_P_ins) tst_someip_provided_instance $(OBJS_NECONFIG) tst_someip_config $(OBJS_network) tst_network $(OBJS_udp) tst_udp \
	$(OBJS_REQ_COMMANDLINE) tst_someip_cmdline_req $(OBJS_PRO_COMMANDLINE) tst_someip_cmdline_pro \
	$(OBJS_PRO1_COMMANDLINE) tst_someip_cmdline_pro1 $(OBJS_REQ1_COMMANDLINE) tst_someip_cmdline_req1
	$(OBJS_PRO_USAGE) tst_someip_pro_usage $(OBJS_REQ_USAGE) tst_someip_req_usage
