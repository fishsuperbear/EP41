# /**

# * Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

# * All Rights Reserved.

# *

# * Redistribution and use in source and binary forms, with or without

# * modification, are NOT permitted except as agreed by

# * iAuto Software(Shanghai) Co., Ltd.

# *

# * Unless required by applicable law or agreed to in writing, software

# * distributed under the License is distributed on an "AS IS" BASIS,

# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# */
CC = g++

CFLAGS = -c -Wall -g
CFLAGS += -D SOMEIP_LINUX_PLATFORM 

LDFLAGS =-lpthread -lm -ldl -std=c++14 -static

LIB =
INCLUDE = -I../../../include -I./ -I../../ -I./include 

all: libsomeip_e2e.a

SRCS = \
	./runtimeE2E/ne_someip_e2e.cpp \
	./src/libcrc/buffer.cpp \
	./src/libcrc/crc.cpp \
	./src/libe2exf/state_machine_dev.cpp \
	./src/libe2exf/transformer.cpp \
	./src/libe2exf/fault_injector.cpp \
	./src/libe2exf/config_reader.cpp \
	./src/libe2exf/e2e_handler.cpp \
	./src/libe2exf/config.cpp \
	./src/libe2exf/dev.cpp \
	./src/libe2e/profile_04.cpp \
	./src/libe2e/checker_11.cpp \
	./src/libe2e/profile_07.cpp \
	./src/libe2e/checker_07.cpp \
	./src/libe2e/checker_05.cpp \
	./src/libe2e/protector_11.cpp \
	./src/libe2e/profile_22.cpp \
	./src/libe2e/checker_06.cpp \
	./src/libe2e/checker_04.cpp \
	./src/libe2e/profile_11.cpp \
	./src/libe2e/result.cpp \
	./src/libe2e/profile_06.cpp \
	./src/libe2e/protector_05.cpp \
	./src/libe2e/protector_04.cpp \
	./src/libe2e/protector_06.cpp \
	./src/libe2e/checker_22.cpp \
	./src/libe2e/protector_07.cpp \
	./src/libe2e/state_machine.cpp \
	./src/libe2e/profile_05.cpp \
	./src/libe2e/protector_22.cpp \
	./e2eProtect/ne_someip_e2e_protect.cpp \

OBJS = $(SRCS:.cpp=.o)

libsomeip_e2e.a:$(OBJS)
	ar cr $@ $^

%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ -c $< -std=c++14

clean:
	rm -rf libsomeip_e2e.a $(OBJS)
