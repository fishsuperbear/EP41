#include <iostream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include "enc_thread.h"


#define SOCKET_NAME "/tmp/.cam_hal_enc_3"
#define SHM_SIZE (1024*1024*2*6) //2M * 6 
#define SHM_KEY_BLOCK0_SENSOR0 0xFF00
#define SHM_KEY_BLOCK0_SENSOR1 0xFF01
#define SHM_KEY_BLOCK0_SENSOR2 0xFF02
#define SHM_KEY_BLOCK0_SENSOR3 0xFF03
#define SHM_KEY_BLOCK1_SENSOR0 0xFF10
#define SHM_KEY_BLOCK1_SENSOR1 0xFF11
#define SHM_KEY_BLOCK1_SENSOR2 0xFF12
#define SHM_KEY_BLOCK1_SENSOR3 0xFF13
#define SHM_KEY_BLOCK2_SENSOR0 0xFF20
#define SHM_KEY_BLOCK2_SENSOR1 0xFF21
#define SHM_KEY_BLOCK2_SENSOR2 0xFF22
#define SHM_KEY_BLOCK2_SENSOR3 0xFF23
#define SHM_KEY_BLOCK3_SENSOR0 0xFF30
#define SHM_KEY_BLOCK3_SENSOR1 0xFF31
#define SHM_KEY_BLOCK3_SENSOR2 0xFF32
#define SHM_KEY_BLOCK3_SENSOR3 0xFF33
#define SHM_KEY_MAX_NUM 12

int main(int argc, char *argv[]) {
    int sensorid = 0;
    if (argc >= 2) {
        if (strcmp(argv[1], "sensorid") == 0) {
            if (argc >= 3) {
                sensorid = atoi(argv[2]);
            }
        }
    }
    char socket_name[128];
    char output_filename[128];
    sprintf(socket_name,"/tmp/.cam_hal_enc_%d",sensorid);
    sprintf(output_filename,"sensor_%d_pid_%d.enc",sensorid,getpid());
    // 创建本地套接字
    int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd < 0) {
        printf("socket error\n");
        exit(1);
    }
    FILE *fp = fopen(output_filename, "wb");
    /* FILE *fp = fopen("output.bin", "wb"); */
    if (fp == NULL) {
        printf("open file error\n");
    }

    // 设置本地套接字地址
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_name, sizeof(addr.sun_path) - 1);
    /* strncpy(addr.sun_path, SOCKET_NAME, sizeof(addr.sun_path) - 1); */

    // 连接到本地套接字
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        printf("connect error\n");
        exit(1);
    }
    printf("socket connect success.\n");
    while(1)
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(sockfd, &readfds);
        struct timeval timeout;
        timeout.tv_sec = 2;  // 设置超时时间为5秒
        timeout.tv_usec = 0;
        int ret = select(sockfd + 1, &readfds, NULL, NULL, &timeout);
        if(ret==0){//timeout
            continue;
        }
        if(ret<0){//exit
            break;
        }


    // 接收服务端发送的命令
    char command[64];
    int read_len = read(sockfd, command, sizeof(command) - 1);
    if (read_len < 0) {
        printf("read error\n");
        exit(1);
    } else if (read_len == 0) {
        printf("connection closed\n");
        exit(1);
    } else {
        command[read_len] = '\0';
        printf("received command: %s\n", command);
    }
    char *token;
    token = strtok(command, ":");
    int data_len = 0;
    char cmd[56];
    if (token != NULL) {
        /* std::cout << "Command: " << token << std::endl; */
        memcpy(cmd,token,strlen(token));
        token = strtok(NULL, ":"); // 继续分割，获取数据长度
        if (token != NULL) {
            data_len = std::stoi(token); // 将字符串转换为整型
            /* std::cout << "Data length: " << data_len << std::endl; */
        } else {
            /* std::cerr << "Invalid command format" << std::endl; */
        }
    }

    // 如果收到的命令是“bufready”，则读取shmbuf并写入文件
    if (strcmp(command, "bufready") == 0) {
        printf("get buf,send unlock\n");
        int shmid = shmget(SHM_KEY_BLOCK0_SENSOR0, SHM_SIZE, 0666|IPC_CREAT);
        if (shmid < 0) {
            printf("shmget error\n");
        }
        void *shmbuf = (void*)shmat(shmid, NULL, 0);
        if (shmbuf == (void*)-1) {
            printf("shmat error\n");
        }
        int write_len = fwrite(shmbuf, 1, data_len, fp);
        if (write_len != data_len) {
            printf("write file error\n");
        }


        // 将共享内存从进程内存中分离
        if (shmdt(shmbuf) < 0) {
            printf("shmdt error\n");
            break;
        }

        // 向服务端发送“unlock”命令
        char unlock[] = "unlock";
        write_len = write(sockfd, unlock, strlen(unlock));
        if (write_len < 0) {
            printf("write error\n");
        }
    }

    }
    // 关闭套接字
    close(sockfd);

    // 关闭文件
    fclose(fp);
    return 0;
}

