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

using namespace std;
///大小端转换
// 4bytes、32bit数据大小端转化
#define L2B32(Little)                                   \
  (((Little & 0xff) << 24) | (((Little)&0xff00) << 8) | \
   (((Little)&0xff0000) >> 8) | ((Little >> 24) & 0xff))
// 2bytes、16bit数据大小端转化
#define L2B16(Little) (((Little & 0xff) << 8) | ((Little >> 8) & 0xff))


#define SERVER_PORT 10000
#define BUFFER_SIZE 100
#define SERVER_QUEUE 20

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

struct Payload {
    uint16_t serviceId;
    union info
    {
      uint16_t instanceNum;
        struct event
        {
          uint16_t groupId;
          uint16_t eventId;
        } event_t;
    } info_t;
};

int main(int argc, char **argv) {
    if (argc < 3 || (strcmp(argv[1], "start") != 0 && strcmp(argv[1], "stop") != 0)) {
      printf("USAGE: %s: start <serviceId> <eventgroup> <eventid>\n", argv[0]);
      printf("USAGE: %s: start <serviceId> <instanceNum>\n", argv[0]);
      printf("USAGE: %s: stop <serviceId> \n", argv[0]);
      return 1;
    }
    printf("start\n");
    int start = !strcmp(argv[1], "start");
    int startEvent = 0;
    uint16_t ServiceId = std::stoi(argv[2]) & 0xFFFF;
    uint16_t InstaceNum = 0;
    uint16_t eventGroupId = 0;
    uint16_t eventId = 0;
    if (start && argc < 4) {
      printf("%s: start <serviceId> <instanceNum>\n", argv[0]);
      return 1;
    }
    if(start && argc >= 5) {
      startEvent = 1;
    }

    if(startEvent) {
      eventGroupId = std::stoi(argv[3]) & 0xFFFF;
      eventId = std::stoi(argv[4]) & 0xFFFF;
    } else if (start) {
      InstaceNum = std::stoi(argv[3]) & 0xFFFF;
    }

    int sockServer = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockServer < 0) {
        printf("socket failed\n");
        exit(1);
    }

    int on = 1;

    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(SERVER_PORT);
    serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
   // serverAddr.sin_addr.s_addr = htonl(0xAC100A32);//172.16.10.50
    //serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);//127.0.0.1

    uint8_t buffer[BUFFER_SIZE] = {0};
    struct UtHeader SomeipUt;
    struct Payload payload;
    memset(buffer, 0, BUFFER_SIZE);
    memset(&payload, 0, sizeof(struct Payload));
    printf("start send message \n");
    SomeipUt.SID = 0x0501;// Big Edien is 0x0105
    SomeipUt.InterfaceVer = 1;
    SomeipUt.ProtcolVer = 1;
    SomeipUt.RID = 0;
    SomeipUt.GID = 127; //0x7F
    payload.serviceId = L2B16(ServiceId);
    if (start) {
      if (startEvent) {
        SomeipUt.PID = 0xF6;
        SomeipUt.Length = L2B32(14);  // 14
        payload.info_t.event_t.groupId = L2B16(eventGroupId);
        payload.info_t.event_t.eventId = L2B16(eventId);
      } else {
        SomeipUt.PID = 0xF8;
        SomeipUt.Length = L2B32(12);  // 12
        payload.info_t.instanceNum = L2B16(InstaceNum);
      }
    } else {
      SomeipUt.PID = 0xF7;
      SomeipUt.Length = L2B32(10);//10
    }
    memcpy(buffer, (char *)&SomeipUt, sizeof(struct UtHeader));
    memcpy(buffer + 16, (void *)(&payload), sizeof(struct Payload));
    printf("hex dump: ");
    for (int i = 0; i < 25; i++) {
      printf("%02X ", buffer[i]);
      if(i == 15) {
        printf(" # ");
      }
    }
    printf("\n");

    printf(
        "send msg of service: 0x%04x, instace: 0x%04x, eventGroupId: 0x%04x, "
        "eventId: 0x%04x\n",
        ServiceId, InstaceNum, eventGroupId, eventId);
    size_t ret = sendto(sockServer, buffer, BUFFER_SIZE, 0, (struct sockaddr *)&serverAddr,
           sizeof(struct sockaddr));
    cout << "sendto with ret=" << ret << endl;
    if(ret == -1) {
      printf("sendto failed. err: %s\n", strerror(errno));
    }
    close(sockServer);
    return 0;
}
/* EOF */
