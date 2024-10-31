/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: doip client socket
 */
#include <iostream>
#include <fstream>
#include <algorithm>
#include <dirent.h>

#include "json/json.h"
#include "common.h"
#include "diag_test_cantp.h"








// #include "isotplib/isotp.h"
// void isotp_user_debug(const char* message, ...)
// {
//     printf("%s", message);
// }
// int32_t isotp_user_send_can(const uint32_t arbitration_id,
//                          const uint8_t* data, const uint8_t size)
// {
//     printf("send: %08x ", arbitration_id);
//     for (int i = 0; i < size; i++)
//         printf("%02x ", data[i]);
//     printf("\n");

//     return ISOTP_RET_OK;
// }
// uint32_t isotp_user_get_ms(void)
// {
//     struct timespec times = {0, 0};
//     long time;

//     clock_gettime(CLOCK_MONOTONIC, &times);
//     time = times.tv_sec * 1000 + times.tv_nsec / 1000000;
//     return time;
// }

void
DiagResqestOnCan::isotp_user_debug(const char* message)
{
    DEBUG_LOG << message;
}

int32_t
DiagResqestOnCan::isotp_user_send_can(const uint32_t arbitration_id,
                        const uint8_t* data, const uint8_t size)
{
    printfVecHex("send", (uint8_t *)data, size);

    int32_t ret;
    int32_t len = size <= 8 ? size : 8;
    ret = socketApi_->SocketCanSend(arbitration_id, (uint8_t *)data, len);
    if (ret < 0) {
        return ISOTP_RET_ERROR;
    }
    return ISOTP_RET_OK;
}

uint32_t
DiagResqestOnCan::isotp_user_get_ms(void)
{
    struct timespec times = {0, 0};
    long time;

    clock_gettime(CLOCK_MONOTONIC, &times);
    time = times.tv_sec * 1000 + times.tv_nsec / 1000000;
    return time;
}









int32_t
DiagResqestOnCan::SocketRawCanSend(uint32_t& canid, std::vector<uint8_t>& data)
{
    printfVecHex("send", &data[0], data.size());
    //size max 8 byte
    int32_t ret;
    int32_t len = data.size() <= 8 ? data.size() : 8;
    ret = socketApi_->SocketCanSend(canid, &data[0], len);
    return ret;
}
int32_t
DiagResqestOnCan::SocketRawCanRecv(uint32_t& canid, std::vector<uint8_t>& data)
{
    //size max 8 byte
    int32_t ret;
    data.resize(8, 0);
    ret = socketApi_->SocketCanRecv(&canid, &data[0], 8, 5000);
    printfVecHex("recv", &data[0], 8);
    return ret;
}

bool
DiagResqestOnCan::CheckResponse(std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch)
{
    if (isExactMatch && respData.size() != srcData.size()) {
        return false;
    }
    if (!isExactMatch && respData.size() < srcData.size()) {
        return false;
    }
    
    for (uint32_t i = 0; i < srcData.size(); i++) {
        if (srcData[i] != respData[i]) {
            return false;
        }
    }

    return true;
}
int32_t
DiagResqestOnCan::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
{
    uint8_t left = 0;
    uint8_t right = 0;

    for (uint32_t index = 0; index < bcd.size(); index++) {
        right = bcd[index];
        if (right >= '0' && right <= '9') {
            right -= '0';
        }
        else if (right >= 'A' && right <= 'F') {
            right -= 'A' - 10;
        }
        else if (right >= 'a' && right <= 'f') {
            right -= 'a' - 10;
        }
        else {
            break;
        }
        
        if (index % 2 == 1) {
            data.push_back(left << 4 | right);
        }
        else {
            left = right;
        }
    }

    return data.size();
}
int32_t
DiagResqestOnCan::ParseJsonFile(const std::string& filename, TestInfoCanTP& testInfo)
{
    Json::Value root;
    Json::CharReaderBuilder reader;
    JSONCPP_STRING errs;

    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }
    
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        std::cerr << "parseJson error! " << errs;
        return -1;
    }
    ifs.close();

    testInfo.request_canid = root["request_canid"].asString();
    testInfo.response_canid = root["response_canid"].asString();
    testInfo.type = root["type"].asString();

    const Json::Value& testItems = root["test"];
    for (const auto& testItem : testItems) {
        TestInfoCanTP::TestItem item;
        const Json::Value& requestArray = testItem["request"];
        for (const auto& request : requestArray) {
            item.request.emplace_back(request.asString());
        }
        const Json::Value& responseArray = testItem["response"];
        for (const auto& response : responseArray) {
            item.response.emplace_back(response.asString());
        }
        item.timeout = testItem["timeout"].asInt();
        item.retryCounts = testItem["retryCounts"].asInt();
        item.delayTime = testItem["delayTime"].asInt();
        item.describe = testItem["describe"].asString();
        testInfo.test.emplace_back(item);
    }

    return 0;
}
int32_t
DiagResqestOnCan::StartTestCanTP()
{
    std::cout << "##Test cantp" << std::endl;

    bool succ = true;
    socketApi_->SocketCanConfig("vcan0", true);

    std::vector<std::string> json_files;
    DIR* dir = opendir("../Conf/docan");

    if (dir) {
        dirent* entry;
        while ((entry = readdir(dir))) {
            std::string filename = entry->d_name;
            if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json") {
                json_files.push_back(filename);
                DEBUG_LOG << "JsonFile :" << filename;
            }
        }
        closedir(dir);
    }
    for(auto file: json_files) {
        ParseJsonFile("../Conf/docan/" + file, testInfo_);
    }
    
    for(auto test_items: testInfo_.test) {
        succ = true;
        
        for(uint32_t i = 0; i < test_items.request.size(); i++) {
            test_items.request[i].erase(std::remove_if(test_items.request[i].begin(), \
                test_items.request[i].end(), ::isspace), \
                test_items.request[i].end() );
            std::vector<uint8_t > vec_request;
            ParseFromBcd(test_items.request[i], vec_request);
            uint32_t canid = std::stoi(testInfo_.request_canid, 0, 16);
            SocketRawCanSend(canid, vec_request);
        }

        for(uint32_t i = 0; i < test_items.response.size(); i++) {
            test_items.response[i].erase(std::remove_if(test_items.response[i].begin(), \
                test_items.response[i].end(), ::isspace), \
                test_items.response[i].end() );
            std::vector<uint8_t > source_resp;
            ParseFromBcd(test_items.response[i], source_resp);
            test_items.isExactMatch = true;
            if (!test_items.response[i].empty() && test_items.response[i].back() == '.') {
                test_items.isExactMatch = false;
            }
            uint32_t canid = std::stoi(testInfo_.request_canid, 0, 16);
            std::vector<uint8_t > recv_resp;
            SocketRawCanRecv(canid, recv_resp);

            int32_t ret = CheckResponse(source_resp, recv_resp, test_items.isExactMatch);
            if (ret) {
                DEBUG_LOG << "check response succ";
            }
            else {
                DEBUG_LOG << "check response fail";
                printfVecHex("sour", &source_resp[0], source_resp.size());
                succ = false;
                break;
            }
        }
        if (succ) {
            PASS_LOG << test_items.describe;
        }
        else {
            FAIL_LOG << test_items.describe;
            break;
        }
    }

    if (!succ) {
        return -1;
    }

    return 0;
}



#if 0
int main() {
    DiagResqestOnCan requestSocket;
    requestSocket.StartTestCanTP();
    return 0;
}
#endif

