/**
 * Copyright @ 2019 iAuto (Shanghai) Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * iAuto (Shanghai) Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <iostream>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <memory>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include "someip/base/log/ne_someip_log.h"
#include "NESomeIPTc8Process.h"


#define DEBUG_LOG(format, ...) \
 { \
    char print_msg[1024]= { 0 };    \
    struct timeval tv;              \
    gettimeofday(&tv, nullptr);     \
    struct tm *timeinfo = localtime(&tv.tv_sec);        \
    uint32_t milliseconds = tv.tv_usec / 1000;          \
    char time_buf[64] = { 0 };                          \
    memset(time_buf, 0x00, sizeof(time_buf));           \
    memset(print_msg, 0x00, sizeof(print_msg));         \
    snprintf(time_buf, sizeof(time_buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d",         \
        timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,             \
        timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec, milliseconds);          \
    snprintf(print_msg, sizeof(print_msg), (format), ##__VA_ARGS__);                   \
    printf("[%s] [%d %ld %s@%s(%d) | %s]\n", time_buf, getpid(), syscall(__NR_gettid), \
        __FUNCTION__, (nullptr == strrchr(__FILE__, '/')) ? __FILE__: (strrchr(__FILE__, '/') + 1), __LINE__, (print_msg)); \
 }


#define SERVER_PORT 10000
#define BUFFER_SIZE 100
#define SERVER_QUEUE 20

static std::shared_ptr<NESomeIPTc8Process> tc8Process = nullptr;

// define the struct for someip message header
struct UtHeader {
    uint16_t SID;         // service id
    uint8_t GID;          // group id
    uint8_t PID;          // service primitive id
    uint32_t Length;      // the length filed of message
    uint32_t dc;          // reserved
    uint8_t ProtcolVer;   // protocol version
    uint8_t InterfaceVer;  // interface version
    uint8_t TID;          // type id
    uint8_t RID;          // result id
};

int SomeipReveiveControl(int socketid, uint8_t *buffer, const struct sockaddr *dest_addr) {
    uint16_t ServiceId, InstaceNum, EventGroup, EventId;
    struct UtHeader SomeipUt;
    memset(&SomeipUt, 0, sizeof(SomeipUt));
    memcpy(&SomeipUt, buffer, sizeof(SomeipUt));
    SomeipUt.SID = ((buffer[0]<< 8) | buffer[1]);
    SomeipUt.Length = ((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | buffer[7]);
    ne_someip_log_debug("SID=0x%x GID=0x%x PID=0x%x Length=0x%x",
        SomeipUt.SID, SomeipUt.GID, SomeipUt.PID, SomeipUt.Length);
    ne_someip_log_debug("ProtcolVer=0x%x InterfaceVer=0x%x TID=0x%x RID=0x%x",
        SomeipUt.ProtcolVer, SomeipUt.InterfaceVer, SomeipUt.TID, SomeipUt.RID);

    if (SomeipUt.SID != 261) {
        ne_someip_log_error("service id error: 0x%x", SomeipUt.SID);
        return -1;
    }

    if (SomeipUt.ProtcolVer != 1) {
        ne_someip_log_error("Protcol Version  error: 0x%x", SomeipUt.ProtcolVer);
        return -1;
    }

    if (SomeipUt.InterfaceVer != 1) {
        ne_someip_log_error("Protcol Version  error: 0x%x", SomeipUt.InterfaceVer);
        return -1;
    }

    if (SomeipUt.RID != 0) {
        ne_someip_log_error("RID  error: 0x%x", SomeipUt.RID);
        return -1;
    }

    ne_someip_log_debug("SomeipUt.GID =0x%x", SomeipUt.GID);
    if (SomeipUt.GID == 127) {
        switch (SomeipUt.PID) {
        case 248: {
            if (SomeipUt.Length == 12) {
                buffer[7] = 8;
                buffer[14] = 128;
                ServiceId = (buffer[16] << 8) | buffer[17];
                InstaceNum = (buffer[18] << 8) | buffer[19];
                ne_someip_log_debug("offer serviec request command receive ,service: 0x%x, instance number: 0x%x",
                    ServiceId, InstaceNum);
                sendto(socketid, buffer, 16, 0, dest_addr, sizeof(struct sockaddr));
                tc8Process->OfferService(ServiceId, InstaceNum);
                return 10;
            } else {
                ne_someip_log_error("Length error: 0x%x", SomeipUt.Length);
                return -1;
            }
        }
        case 247: {
            if (SomeipUt.Length == 10) {
                buffer[7] = 8;
                buffer[14] = 128;
                ServiceId = (buffer[16] << 8) | buffer[17];

                tc8Process->StopService(ServiceId);
                ne_someip_log_debug("stop serviec request command receive ,service: 0x%x", ServiceId);
                sendto(socketid, buffer, 16, 0, dest_addr, sizeof(struct sockaddr));
                return 20;
            } else {
                ne_someip_log_debug("Length error: 0x%x", SomeipUt.Length);
                return -1;
            }
        }
        case 246: {
            if (SomeipUt.Length == 14) {
                buffer[7] = 8;
                buffer[14] = 128;

                ServiceId = (buffer[16] << 8) | buffer[17];
                EventGroup = (buffer[18] << 8) | buffer[19];
                EventId = (buffer[20] << 8) | buffer[21];
                sendto(socketid, buffer, 16, 0, dest_addr, sizeof(struct sockaddr));

                tc8Process->TriggerEvent(ServiceId, EventGroup, EventId);
                ne_someip_log_debug("event trigge event request command receive ,service: 0x%x", ServiceId);
                ne_someip_log_debug("Eventgrpoup: 0x%x, eventid : 0x%x", EventGroup, EventId);
                return 30;
            } else {
                ne_someip_log_error("Length error: 0x%x", SomeipUt.Length);
                return -1;
            }
        }
        default: {
            ne_someip_log_error("PID error: 0x%x", SomeipUt.PID);
            return -1;
        }
        }
    } else if (SomeipUt.GID == 0) {
        if (SomeipUt.PID == 2) {
            if (SomeipUt.Length == 8) {
                buffer[14] = 128;
                ne_someip_log_debug("start service request command");
                int ret = sendto(socketid, buffer, 16, 0, dest_addr, sizeof(struct sockaddr));
                if (ret < 0) {
                    ne_someip_log_error("sendto error [%s]", strerror(errno));
                }
                ne_someip_log_debug("sendto success [%d]", ret);
                return 40;
            } else {
                ne_someip_log_error("Length error: 0x%x", SomeipUt.Length);
                return -1;
            }
        } else if (SomeipUt.PID == 3) {
            ne_someip_log_debug("SomeipUt.Length = 0x%x", SomeipUt.Length);
            buffer[7] = 8;
            buffer[14] = 128;
            ne_someip_log_debug("End service request command");
            sendto(socketid, buffer, 16, 0, dest_addr, sizeof(struct sockaddr));
            return 50;
        }
    } else {
        ne_someip_log_error("GID erro 0x%x", SomeipUt.GID);
        return -1;
    }
    return 0;
}

int main() {

    ne_someip_log_init("someip_ets", 1, 0, "/opt/usr/log/soc/", 10,  10*1024*1024);

    DEBUG_LOG("start");
    int sockServer = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockServer < 0) {
        DEBUG_LOG("socket failed");
        exit(1);
    }

    int on = 1;
    if (setsockopt(sockServer, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on))) {
        DEBUG_LOG("setsockopt SO_REUSEADDR error, %s ", strerror(errno));
        close(sockServer);
        exit(1);
    }

    if (setsockopt(sockServer, SOL_SOCKET, SO_REUSEPORT, (char*)(&on), sizeof(on))) {
        DEBUG_LOG("setsockopt SO_REUSEPORT error, %s ", strerror(errno));
        close(sockServer);
        exit(1);
    }

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (::bind(sockServer, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        DEBUG_LOG("sock_server::main, Bind failed %s ", strerror(errno));
        exit(1);
    }

    struct sockaddr_in clientAddr;
    memset(&clientAddr, 0, sizeof(clientAddr));
    socklen_t length = sizeof(clientAddr);

    tc8Process = std::make_shared<NESomeIPTc8Process>();

    DEBUG_LOG("init() start ");
    bool ret = tc8Process->init();
    if (!ret) {
        DEBUG_LOG("init() failed ");
        exit(1);
    }
    DEBUG_LOG("init() success ");

    uint8_t buffer[BUFFER_SIZE] = { 0 };
    DEBUG_LOG("start recvfrom message ");
    while (1) {
        memset(buffer, 0, sizeof(buffer));
        int n = recvfrom(sockServer, buffer, sizeof(buffer), 0, (struct sockaddr *)&clientAddr, &length);
        DEBUG_LOG("recvfrom message length [%d] ", n);
        if (n > 25) {
            char testtest[100] = { 0 };
            memcpy(testtest, buffer + 25, n - 25);
            DEBUG_LOG(" **** [%s] **** ", testtest);
        }
        DEBUG_LOG("ip : %d, port : %d", clientAddr.sin_addr.s_addr, ntohs(((struct sockaddr_in*)&clientAddr)->sin_port));
        SomeipReveiveControl(sockServer, buffer, (struct sockaddr *)&clientAddr);
    }
    close(sockServer);

    ne_someip_log_deinit();
    return 0;
}
/* EOF */
