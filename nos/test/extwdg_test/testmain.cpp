#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <map>
#include <thread>

static const std::map<std::uint32_t, std::uint32_t> qa_table = 
{
    {0x0,0xFF0FF000},
    {0x1,0xB040BF4F},
    {0x2,0xE919E616},
    {0x3,0xA656A959},
    {0x4,0x75857A8A},
    {0x5,0x3ACA35C5},
    {0x6,0x63936C9C},
    {0x7,0x2CDC23D3},
    {0x8,0xD222DD2D},
    {0x9,0x9D6D9262},
    {0xA,0xC434CB3B},
    {0xB,0x8B7B8474},
    {0xC,0x58A857A7},
    {0xD,0x17E718E8},
    {0xE,0x4EBE41B1},
    {0xF,0x01F10EFE}
};


struct MessageBuffer
{
    uint32_t head = 0xFACE;
    uint8_t seq = 0;
    uint8_t request = 0;
    uint32_t data = 0;
};



int main() {
    // 创建UDP套接字
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("socket");
        return 1;
    }

    // 设置服务器地址和端口
    struct sockaddr_in serverAddr{};
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(23462);
    if (inet_pton(AF_INET, "172.16.90.11", &(serverAddr.sin_addr)) <= 0) {
        perror("inet_pton");
        return 1;
    }
    if(bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        perror("Failed to bind socket");
        return 1;
    }
    struct sockaddr_in clientAddr{};
    clientAddr.sin_family = AF_INET;
    clientAddr.sin_port = htons(23461);
    if (inet_pton(AF_INET, "172.16.90.11", &(clientAddr.sin_addr)) <= 0) {
        perror("inet_pton");
        return 1;
    }
    MessageBuffer test_send, recv_buff;
    // 接收服务器回复
    std::thread t1 = std::thread([&](){
        while(1)
        {
            std::cout << "Recv loop enter!"<<std::endl;;
            memset(&recv_buff, 0, sizeof(recv_buff));
            ssize_t recvBytes = recvfrom(sockfd, &recv_buff, sizeof(recv_buff), 0, nullptr, nullptr);
            if (recvBytes < 0) {
                perror("recvfrom");
                return 1;
            }
            // 打印服务器回复
            std::cout << " recv seq :" << recv_buff.seq << " recv data :"<< recv_buff.data <<std::endl;
        }
    });
    // 无限循环等待用户输入
    while (true) {
        // 提示用户输入消息
        std::cout << "please input the message and enter exit to quit" << std::endl;
        std::string message;
        std::getline(std::cin, message);

        // 检查是否退出程序
        if (message == "exit") {
            break;
        }
        if (message == "1") {
            test_send.request = 0x3;
            test_send.data = 0x1;
        }

        if (message == "2") {
            test_send.request = 0x1;
            test_send.data = 0xFFFF;
        }

        if (message == "3") {
            test_send.request = 0x2;
            test_send.data = 0xFFFF;
        }
        
        if (message == "4") {
            test_send.request = 0x3;
            test_send.data = 0x5;
        }

        if (message == "5") {
            test_send.request = 0x3;
            test_send.data = 0x8;
        }

        if (message == "6") {
            test_send.request = 0x3;
            test_send.data = 0xb;
        }


        if(message != "1"&& message != "2"&& message != "3"&& message != "4"&& message != "5"&& message != "6")
        {
            continue;
        }

        // 发送数据到服务器
        std::cout << "send data :" << message << std::endl;
        ssize_t sentBytes = sendto(sockfd, &test_send, sizeof(test_send), 0, (struct sockaddr*)&clientAddr, sizeof(clientAddr));
        if (sentBytes < 0) {
            perror("sendto");
            return 1;
        }


    }
    t1.join();
    // 关闭套接字
    close(sockfd);

    return 0;
}