CC = g++

CFLAGS = -c -Wall -g -fPIC -std=c++14
CFLAGS += -D SOMEIP_LINUX_PLATFORM
CFLAGS += -D SOMEIP_PC_PLATFORM
LDFLAGS =-lpthread -lm -ldl -std=c++14 -lcrypto -lssl -L../tests/openssl-1.1.1i

PROJ_ROOT_DIR = ..
PROJ_E2E_DIR = ../src/extend/e2e
LIBSOMEIP = $(PROJ_ROOT_DIR)/test_e2e/libsomeip/libsomeip.so

INCLUDE_DIRS += -I$(PROJ_E2E_DIR)
INCLUDE_DIRS += -I$(PROJ_E2E_DIR)/include
INCLUDE_DIRS += -I$(PROJ_ROOT_DIR)/include
INCLUDE_DIRS += -I$(PROJ_ROOT_DIR)/src/base/log
INCLUDE_DIRS += -I$(PROJ_ROOT_DIR)/src/base/config

TARGET = libsomeip_e2e.so someip_server someip_client
all:$(TARGET)

################################################################################################################
# libsomeip_e2e.so
LIBSOMEIP_E2E_FILES= \
	$(PROJ_E2E_DIR)/src/libcrc/buffer.cpp \
	$(PROJ_E2E_DIR)/src/libcrc/crc.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_04.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_05.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_06.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_07.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_11.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/checker_22.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_04.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_05.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_06.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_07.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_11.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/profile_22.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_04.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_05.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_06.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_07.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_11.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/protector_22.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/result.cpp \
	$(PROJ_E2E_DIR)/src/libe2e/state_machine.cpp \
	$(PROJ_E2E_DIR)/runtimeE2E/NESomeIPE2EManager.cpp \
	$(PROJ_E2E_DIR)/runtimeE2E/NESomeIPE2ECheckerImpl.cpp \
	$(PROJ_E2E_DIR)/runtimeE2E/NESomeIPE2EManagerImpl.cpp \
	$(PROJ_E2E_DIR)/runtimeE2E/NESomeIPE2EProtectorImpl.cpp

LIBSOMEIP_E2E_OBJ=$(LIBSOMEIP_E2E_FILES:.cpp=.o)

libsomeip_e2e.so:$(LIBSOMEIP_E2E_OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) -shared $(LIBSOMEIP)

################################################################################################################
# someip_server
SOMEIP_SERVER_FILES = ./someip_server.cpp

SOMEIP_SERVER_OBJ = $(SOMEIP_SERVER_FILES:.cpp=.o)

someip_server:$(SOMEIP_SERVER_OBJ)
	$(CC) -o $@ $(SOMEIP_SERVER_OBJ) $(LDFLAGS) $(LIBSOMEIP) libsomeip_e2e.so

################################################################################################################
# someip_client
SOMEIP_CLIENT_FILES = ./someip_client.cpp

SOMEIP_CLIENT_OBJ = $(SOMEIP_CLIENT_FILES:.cpp=.o)

someip_client:$(SOMEIP_CLIENT_OBJ)
	$(CC) -o $@ $(SOMEIP_CLIENT_OBJ) $(LDFLAGS) $(LIBSOMEIP) libsomeip_e2e.so

################################################################################################################

%.o:%.cpp
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -o $@ -c $<

################################################################################################################

clean:
	rm -rf $(TARGET) $(LIBSOMEIP_E2E_OBJ) $(SOMEIP_SERVER_OBJ) $(SOMEIP_CLIENT_OBJ)