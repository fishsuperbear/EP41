/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: doip client test manager
 */
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <memory>
#include <fstream>
#include <algorithm>
#include <dirent.h>

#include "diag/diag_sa/include/security_algorithm.h"
#include "json/json.h"
#include "common.h"
#include "uds_request_oncan.h"

using namespace hozon::netaos::diag;



bool
UdsResqestFuncOnCan::CompareResponseData(std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch)
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
UdsResqestFuncOnCan::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
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
UdsResqestFuncOnCan::ParseJsonFile(std::string file)
{
    //解析json文件放到requestTestInfo_
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (!ifs.is_open()) {
        FAIL_LOG << "open json file failed.";
        return -1;
    }

    Json::CharReaderBuilder reader;
    Json::Value root, jsonRequest;
    JSONCPP_STRING errs;

    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        FAIL_LOG << "parseJson error! " << errs;
        return -1;
    }

    DoipRequestInfo_t requestInfo;
    std::string strValue;
    requestInfo.sourceAddress = root["sourceAddress"].asInt();
    requestInfo.targetAddress = root["targetAddress"].asInt();
    strValue = root["canid"].asString();
    requestInfo.canid = std::stoi(strValue, 0, 16);
    requestInfo.canfname = root["canfname"].asString();
    requestInfo.retryCounts = root["retryCounts"].asInt();
    requestInfo.delayTimeMs = root["delayTime"].asInt();
    jsonRequest = root["requestTest"];

    for (uint32_t i = 0; i < jsonRequest.size(); i++) {
        requestInfo.describe = jsonRequest[i]["describe"].asString();
        DEBUG_LOG << "describe: " << requestInfo.describe;
        udsRequestInfo_.push_back(requestInfo);
        
        Json::Value diagResquestInfo;
        diagResquestInfo = jsonRequest[i]["requestTest2"];
        for (uint32_t j = 0; j < diagResquestInfo.size(); j++) {
            DiagRequestInfo_t diagRequestInfo;
            
            diagRequestInfo.sourceAddress = requestInfo.sourceAddress;
            diagRequestInfo.targetAddress = requestInfo.targetAddress;
            strValue = diagResquestInfo[j]["requestType"].asString();
            diagRequestInfo.type = "PhysicAddr" == strValue ? DOIP_TA_TYPE_PHYSICAL
                        : "FunctionAddr" == strValue ? DOIP_TA_TYPE_FUNCTIONAL
                        : DOIP_TA_TYPE_PHYSICAL;

            diagRequestInfo.requestData.clear();
            diagRequestInfo.responseData.clear();
            strValue.clear();
            strValue = diagResquestInfo[j]["request"].asString();
            DEBUG_LOG << "request : " << strValue;
            strValue.erase(std::remove_if(strValue.begin(), strValue.end(), ::isspace), strValue.end());
            ParseFromBcd(strValue, diagRequestInfo.requestData);
            strValue.clear();
            strValue = diagResquestInfo[j]["response"].asString();
            DEBUG_LOG << "response: " << strValue;
            strValue.erase(std::remove_if(strValue.begin(), strValue.end(), ::isspace), strValue.end());
            ParseFromBcd(strValue, diagRequestInfo.responseData);
            diagRequestInfo.isExactMatch = true;
            if (!strValue.empty() && strValue.back() == '.') {
                diagRequestInfo.isExactMatch = false;
            }
            DEBUG_LOG << "isExactMatch: " << diagRequestInfo.isExactMatch;
            diagRequestInfo.timeoutMs = diagResquestInfo[j]["timeout"].asInt();
            diagRequestInfo.timeoutMs = diagRequestInfo.timeoutMs > 0 ? diagRequestInfo.timeoutMs : 5000;
            DEBUG_LOG << "timeoutMs: " << diagRequestInfo.timeoutMs;

            diagRequestInfo.retryCounts = diagResquestInfo[j]["retryCounts"].asInt();
            DEBUG_LOG << "retryCounts: " << diagRequestInfo.retryCounts;
            diagRequestInfo.retryCounts = diagRequestInfo.retryCounts > 0 ? diagRequestInfo.retryCounts : 3;
            diagRequestInfo.delayTimeMs = diagResquestInfo[j]["delayTime"].asInt();
            DEBUG_LOG << "delayTimeMs: " << diagRequestInfo.delayTimeMs;
            DEBUG_LOG << "===============";

            udsRequestInfo_.back().vecRequest.push_back(diagRequestInfo);
        }
        
    }

    ifs.close();
    return 0;
}



void
UdsResqestFuncOnCan::isotp_user_debug(const char* message)
{
    DEBUG_LOG << message;
}
int32_t
UdsResqestFuncOnCan::isotp_user_send_can(const uint32_t arbitration_id,
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
UdsResqestFuncOnCan::isotp_user_get_ms(void)
{
    struct timespec times = {0, 0};
    long time;

    clock_gettime(CLOCK_MONOTONIC, &times);
    time = times.tv_sec * 1000 + times.tv_nsec / 1000000;
    return time;
}

int32_t
UdsResqestFuncOnCan::StartTestUdsOnCan()
{
    std::cout << "##uds on can" << std::endl;

    int32_t ret;
    uint32_t canid;
    uint8_t recv_buf[8];
    bool succ = true;
    
    std::vector<std::string> json_files;
    DIR* dir = opendir("../Conf/uds");
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
        ParseJsonFile("../Conf/uds/" + file);
    }

    for (auto doipInfo : udsRequestInfo_) {
        if (!doipInfo.canfname.empty()) {
            canfname_ = doipInfo.canfname;
            break;
        }
    }
    for (auto doipInfo : udsRequestInfo_) {
        if (doipInfo.canid != 0) {
            canid_ = doipInfo.canid;
            break;
        }
    }

    DEBUG_LOG << "can ifname: " << canfname_;
    if (socketApi_->SocketCanConfig(canfname_.c_str(), true) < 0) {
        return -1;
    }
    for(auto request: udsRequestInfo_) {
        
        succ = true;
        for(auto data: request.vecRequest) {
            
            //1. send
            if(g_link.send_status == ISOTP_SEND_STATUS_IDLE
                && g_link.receive_status == ISOTP_RECEIVE_STATUS_IDLE) {
                printfVecHex("=====send", &data.requestData[0], data.requestData.size());
                isotp_->isotp_send(&g_link, &data.requestData[0], data.requestData.size());
            }

            // handle multiple packet
            while(1) {
                //2. poll time
                uint32_t timeout = 0x7fffffff;
                isotp_->isotp_poll(&g_link);
                if(g_link.send_status == ISOTP_SEND_STATUS_INPROGRESS) {
                    timeout = g_link.send_st_min;
                    if(g_link.send_sn == 1) {
                        timeout = 100;
                    }
                    DEBUG_LOG << "timeout " << timeout;
                }
                
                //3. block recv
                memset(recv_buf, 0, sizeof(recv_buf));
                ret = socketApi_->SocketCanRecv(&canid, recv_buf, 8, timeout);
                if (ret > 0) {
                    printfVecHex("recv", recv_buf, 8);
                    isotp_->isotp_on_can_message(&g_link, recv_buf, 8);
                }
                else {
                    if (g_link.receive_status == ISOTP_RECEIVE_STATUS_FULL &&
                        g_link.send_status != ISOTP_SEND_STATUS_INPROGRESS) {
                        break;
                    }
                }
                DEBUG_LOG << "status " << (int32_t)g_link.receive_status << " "<< (int32_t)g_link.send_status;
                
                //4. check recv end
                if (g_link.receive_status == ISOTP_RECEIVE_STATUS_FULL) {
                    uint16_t out_size;
                    uint8_t payload[100];
                    isotp_->isotp_receive(&g_link, payload, sizeof(payload), &out_size);
                    printfVecHex("=====recv", payload, out_size);
                    
                    std::vector<uint8_t> vec_recv(payload, payload + out_size);
                    CompareResponseData(data.responseData, vec_recv, data.isExactMatch);
                    if (CompareResponseData(data.responseData, vec_recv, data.isExactMatch)) {
                        DEBUG_LOG << "response ok.";
                        if (out_size >= 6 && payload[0] == 0x67 && 
                            (payload[1] == 0x03 || payload[1] == 0x05 || payload[1] == 0x11)) {
                            seed_ = (uint8_t)(payload[2]) << 24 | (uint8_t)(payload[3]) << 16
                                    | (uint8_t)(payload[4]) << 8 | (uint8_t)(payload[5]);
                        }
                    }
                    else {
                        DEBUG_LOG << "response error.";
                        succ = false;
                    }

                    break;
                }
            }
            if (!succ) {
                break;
            }
        }
        if (succ) {
            PASS_LOG << request.describe;
        }
        else {
            FAIL_LOG << request.describe;
            break;
        }
    }

    if (!succ) {
        return -1;
    }

    return 0;
}


#if 0
g++ uds_request_oncan.cpp ../../Common/socketraw/socket.cpp ~/git_hozon/netaos/middleware/diag/diag_sa/src/security_algorithm/security_algorithm.cpp \
-I../../../../netaos_thirdparty/x86/jsoncpp/include \
-I../../../../middleware \
-I../../Common -ljsoncpp \
-L../../../../netaos_thirdparty/x86/jsoncpp/lib
int main() {
    UdsResqestFuncOnCan udsRequest;
    udsRequest.StartTestUdsOnCan();
    return 0;
}
#endif


