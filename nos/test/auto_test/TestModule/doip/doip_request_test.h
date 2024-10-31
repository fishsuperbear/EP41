/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#ifndef DOIP_AUTO_TEST_H
#define DOIP_AUTO_TEST_H


//
#include <string>
#include <vector>

typedef enum DOIP_MESSAGE_TYPE
{
    MESG_HEAD_NEGATIVE_ACK       = 0x0000,
    MESG_ACTIVATE_REQUEST5       = 0x0005,
    MESG_ACTIVATE_RESPONSE       = 0x0006,
    MESG_ACTIVATE_RESPONS1       = 0x0007,
    MESG_ACTIVATE_RESPONS2       = 0x0008,
    MESG_DIAGNOSTIC_MESSAGE      = 0x8001,
    MESG_DIAGNOSTIC_POSITIVE     = 0x8002,
    MESG_DIAGNOSTIC_NEGATIVE     = 0x8003
} doip_message_type_t;


struct TestItem {
    std::string protocol_type;
    int32_t timeout;
    int32_t retryCounts;
    bool isExactMatch;
    
    struct DoipRequest {
        std::string version;
        std::string ver_inverse;
        std::string payload_type;
        int32_t payload_len;
        std::string source_addr;
        std::string target_addr;
        std::string payload_data;
    } ;
    struct DoipResponse {
        std::string version;
        std::string ver_inverse;
        std::string payload_type;
        int32_t payload_len;
        std::string source_addr;
        std::string target_addr;
        std::string payload_data;
        bool isExactMatch;
    } ;

    std::vector<DoipRequest> doip_request;
    std::vector<DoipResponse> doip_response;
    std::string describe;
    int32_t delay_timems;
    bool ignoreFail;
    bool tcp_close;
    uint32_t sever_close_wait;
};
struct TestInfo {
    std::string ip_addr;
    int32_t port;
    std::string ifname;
    std::vector<TestItem> test_items;
    int32_t thread_delay;// thread delay(ms) befor start.
};

class DoipRequestTest {
public:
    DoipRequestTest() {
    }
    ~DoipRequestTest() {
    }
    int32_t StartTestDoip();

private:
    int32_t ParseJsonFile(const std::string& filename, TestInfo& info);
    int32_t ParseJsonFile2(const std::string& filename, std::vector<TestInfo>& info);

    TestInfo doipRequestInfo_;
    std::vector<std::vector<TestInfo>> doipRequestInfo2_;
};


#endif

