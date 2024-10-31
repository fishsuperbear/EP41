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
#include "uds_request_onip.h"

using namespace hozon::netaos::diag;



bool
UdsResqestFuncOnIP::CompareResponseData(std::vector<uint8_t > srcData, std::vector<uint8_t > respData, bool isExactMatch)
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
UdsResqestFuncOnIP::Ipv4TcpDoipPayloadSend(doip_payload_t *doip_payload)
{
    uint8_t send_data[1000];
    int32_t index = 0;
    uint32_t payload_length;
    uint16_t addr;

    memset(send_data, 0, sizeof(send_data));
    send_data[index++] = doip_payload->version;
    send_data[index++] = doip_payload->ver_inverse;
    addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->payload_type);
    memcpy(send_data + index, &addr, 2);
    index += 2;

    payload_length = LITTLE_TO_BIG_ENDIAN_32(doip_payload->payload_len);
    memcpy(send_data + index, &payload_length, 4);
    index += 4;
    addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->source_addr);
    memcpy(send_data + index, &addr, 2);
    index += 2;
    addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->target_addr);
    memcpy(send_data + index, &addr, 2);
    index += 2;
    memcpy(send_data + index, doip_payload->data, doip_payload->payload_len - 4);
    index += (doip_payload->payload_len - 4);

    printfVecHex("send:", send_data, index);
    socketApi_->Ipv4TcpSendData(send_data, index);

    return index;
}

int32_t
UdsResqestFuncOnIP::Ipv4TcpDoipPayloadRecv(doip_payload_t *response, uint32_t timeoutMs)
{
    int ret;
    uint8_t recv_data[4096];

    memset(recv_data, 0, sizeof(recv_data));
    ret = socketApi_->Ipv4TcpRecvData(recv_data, 8, timeoutMs);
    if (ret != 8) {
        socketApi_->Ipv4TcpClose();
        FAIL_LOG << "recv tcp data error, ret " << ret;
        return -1;
    }

    int32_t index = 0;
    response->version = recv_data[index++];
    response->ver_inverse = recv_data[index++];
    response->payload_type = (recv_data[index] << 8) | recv_data[index + 1];
    index += 2;
    response->payload_len = (recv_data[index] << 24) | (recv_data[index + 1] << 16) | (recv_data[index + 2] << 8) | recv_data[index + 3];
    index += 4;

    if (sizeof(recv_data) < response->payload_len + index) {
        socketApi_->Ipv4TcpClose();
        FAIL_LOG << "sizeof recv_buf error, payload_len " << response->payload_len + index;
        return -1;
    }
    ret = socketApi_->Ipv4TcpRecvData(recv_data + index, response->payload_len, timeoutMs);
    if (ret != (int32_t)response->payload_len) {
        socketApi_->Ipv4TcpClose();
        FAIL_LOG << "recv tcp data error2, ret " << ret;
        return -1;
    }
    if (response->payload_len < 4) {
        response->data = new uint8_t[response->payload_len]();
        memcpy(response->data, recv_data + index, response->payload_len);
        index += response->payload_len;
        printfVecHex("recv:", recv_data, index);
        return index;
    }
    
    response->source_addr = (recv_data[index] << 8) | recv_data[index + 1];
    index += 2;
    response->target_addr = (recv_data[index] << 8) | recv_data[index + 1];
    index += 2;
    response->data = new uint8_t[response->payload_len - 4]();
    memcpy(response->data, recv_data + index, response->payload_len - 4);
    index += (response->payload_len - 4);

    printfVecHex("recv:", recv_data, index);
    return index;
}

int32_t
UdsResqestFuncOnIP::Ipv4TcpDoipActivate(doip_request_t *requestInfo, uint32_t timeoutMs)
{
    int32_t ret = socketApi_->Ipv4TcpConnect(ip_.c_str(), port_, ifName_.c_str());
    if (ret < 0) {
        return -1;
    }
    
    doip_payload_t doip_payload;
    doip_payload.version = 0x02;
    doip_payload.ver_inverse = 0xFD;
    doip_payload.payload_type = 0x0005;
    doip_payload.payload_len = 4+3;
    doip_payload.source_addr = requestInfo->source_addr;
    doip_payload.target_addr = requestInfo->target_addr;
    doip_payload.data = new uint8_t[3]();
    //memcpy(doip_payload.data, requestInfo.data, 3);//default payload 000, needn't memcpy
    
    ret = Ipv4TcpDoipPayloadSend(&doip_payload);
    if (ret < 0) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    delete[] doip_payload.data;
    
    doip_payload.data = NULL;
    ret = Ipv4TcpDoipPayloadRecv(&doip_payload, timeoutMs);
    if (ret < 0) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    if (doip_payload.payload_type != 0x0006) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    
    return ret;
}

int32_t
UdsResqestFuncOnIP::Ipv4TcpDoipRequestUds(doip_request_t *requestInfo, doip_request_t *response, uint32_t timeoutMs)
{
    int32_t ret;
    
    doip_payload_t doip_payload;
    doip_payload.version = 0x02;
    doip_payload.ver_inverse = 0xFD;
    doip_payload.payload_type = 0x8001;
    doip_payload.payload_len = 4 + requestInfo->data_length;
    doip_payload.source_addr = requestInfo->source_addr;
    doip_payload.target_addr = requestInfo->target_addr;
    doip_payload.data = new uint8_t[requestInfo->data_length]();
    memcpy(doip_payload.data, requestInfo->data, requestInfo->data_length);
    
    ret = Ipv4TcpDoipPayloadSend(&doip_payload);
    if (ret < 0) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    delete[] doip_payload.data;
    
    doip_payload.data = NULL;
    ret = Ipv4TcpDoipPayloadRecv(&doip_payload, timeoutMs);
    if (ret < 0) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    if (doip_payload.payload_type != 0x8002) {
        socketApi_->Ipv4TcpClose();
        delete[] doip_payload.data;
        return -1;
    }
    delete[] doip_payload.data;
    
    while (1) {
        doip_payload.data = NULL;
        ret = Ipv4TcpDoipPayloadRecv(&doip_payload, timeoutMs);
        if (ret < 0) {
            socketApi_->Ipv4TcpClose();
            delete[] doip_payload.data;
            return -1;
        }
        if (doip_payload.payload_type != 0x8001) {
            socketApi_->Ipv4TcpClose();
            delete[] doip_payload.data;
            return -1;
        }
        if (doip_payload.data[0] == 0x7f && doip_payload.data[2] == 0x78) {
            delete[] doip_payload.data;
            continue;
        }
        break;
    }
    
    response->source_addr = doip_payload.source_addr;
    response->target_addr = doip_payload.target_addr;
    response->data_length = doip_payload.payload_len - 4;
    response->data = new uint8_t[response->data_length]();
    memcpy(response->data, doip_payload.data, response->data_length);
    delete[] doip_payload.data;
    
    return ret;
}

int32_t
UdsResqestFuncOnIP::DoipRequestUDSdata(DiagRequestInfo_t& requestInfo, bool ignoreKey)
{
    int32_t ret;
    doip_request_t request, response;
    uint32_t timeout = requestInfo.timeoutMs;

    do {
        if (!active_status_) {
            doip_request_t activate;
            activate.source_addr = requestInfo.sourceAddress;
            activate.target_addr = 0;
            activate.data = new uint8_t[3]();
            activate.data_length = 3;
            ret = Ipv4TcpDoipActivate(&activate, timeout);
            if (ret < 0) {
                delete[] activate.data;
                break;
            }
            active_status_ = true;
            delete[] activate.data;
        }

        if (!ignoreKey) {
            if (requestInfo.requestData.size() == 2 &&
                requestInfo.requestData[0] == 0x27 && requestInfo.requestData[1] == 0x12) {
                uint32_t key = SecurityAlgorithm::Instance()->RequestSecurityAlgorithm(SecurityAlgorithm::SecurityLevel_FBL_Key, 0xAB854A17, seed_);
                seed_ = 0;
                requestInfo.requestData.push_back((uint8_t)(key >> 24));
                requestInfo.requestData.push_back((uint8_t)(key >> 16));
                requestInfo.requestData.push_back((uint8_t)(key >> 8));
                requestInfo.requestData.push_back((uint8_t)(key));
                DEBUG_LOG << "key1: " << key;
            }
            else if (requestInfo.requestData.size() == 2 &&
                requestInfo.requestData[0] == 0x27 && requestInfo.requestData[1] == 0x04) {
                uint32_t key = SecurityAlgorithm::Instance()->RequestSecurityAlgorithm(SecurityAlgorithm::SecurityLevel_1_Key, 0x23AEBEFD, seed_);
                seed_ = 0;
                requestInfo.requestData.push_back((uint8_t)(key >> 24));
                requestInfo.requestData.push_back((uint8_t)(key >> 16));
                requestInfo.requestData.push_back((uint8_t)(key >> 8));
                requestInfo.requestData.push_back((uint8_t)(key));
                DEBUG_LOG << "key2: " << key;
            }
            else if (requestInfo.requestData.size() == 2 &&
                requestInfo.requestData[0] == 0x27 && requestInfo.requestData[1] == 0x06) {
                uint32_t key = SecurityAlgorithm::Instance()->RequestSecurityAlgorithm(SecurityAlgorithm::SecurityLevel_TEST_Key, 0x1F2E3D4C, seed_);
                seed_ = 0;
                requestInfo.requestData.push_back((uint8_t)(key >> 24));
                requestInfo.requestData.push_back((uint8_t)(key >> 16));
                requestInfo.requestData.push_back((uint8_t)(key >> 8));
                requestInfo.requestData.push_back((uint8_t)(key));
                DEBUG_LOG << "keyT: " << key;
            }
        }


        request.source_addr = requestInfo.sourceAddress;
        request.target_addr = requestInfo.targetAddress;
        request.ta_type = requestInfo.type;
        request.data = new uint8_t[requestInfo.requestData.size()]();
        memcpy(request.data, &requestInfo.requestData[0], requestInfo.requestData.size());
        request.data_length = requestInfo.requestData.size();
        response.data = NULL;
        ret = Ipv4TcpDoipRequestUds(&request, &response, timeout);
        if (ret < 0) {
            delete[] request.data;
            break;
        }
        delete[] request.data;

        std::vector<uint8_t > respData = {response.data, response.data + response.data_length};
        if (CompareResponseData(requestInfo.responseData, respData, requestInfo.isExactMatch)) {
            if (response.data_length >= 6 && response.data[0] == 0x67 && 
                (response.data[1] == 0x03 || response.data[1] == 0x05 || response.data[1] == 0x11)) {
                seed_ = (uint8_t)(response.data[2]) << 24 | (uint8_t)(response.data[3]) << 16
                        | (uint8_t)(response.data[4]) << 8 | (uint8_t)(response.data[5]);
            }
            DEBUG_LOG << "response ok.";
            ret = 0;
        }
        else {
            DEBUG_LOG << "response error.";
            ret = -1;
        }

    } while(0);


    delete[] response.data;
    return ret;
}

int32_t
UdsResqestFuncOnIP::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
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
UdsResqestFuncOnIP::ParseJsonFile(std::string file)
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
        FAIL_LOG << "parseJson error! code: " << errs;
        return -1;
    }

    DoipRequestInfo_t requestInfo;
    requestInfo.ip = root["ip"].asString();
    requestInfo.port = root["port"].asInt();
    requestInfo.ifname = root["ifname"].asString();
    requestInfo.sourceAddress = root["sourceAddress"].asInt();
    requestInfo.targetAddress = root["targetAddress"].asInt();
    requestInfo.retryCounts = root["retryCounts"].asInt();
    requestInfo.delayTimeMs = root["delayTime"].asInt();
    jsonRequest = root["requestTest"];

    for (uint32_t i = 0; i < jsonRequest.size(); i++) {

        requestInfo.describe = jsonRequest[i]["describe"].asString();
        DEBUG_LOG << "describe: " << requestInfo.describe;
        requestInfo.ignoreFail = jsonRequest[i].get("ignoreFail", false).asBool();
        requestInfo.ignoreKey = jsonRequest[i].get("ignoreKey", false).asBool();
        doipRequestInfo_.push_back(requestInfo);
        
        Json::Value diagResquestInfo;
        diagResquestInfo = jsonRequest[i]["requestTest2"];
        for (uint32_t j = 0; j < diagResquestInfo.size(); j++) {
            DiagRequestInfo_t diagRequestInfo;
            std::string strValue;
            
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


            doipRequestInfo_.back().vecRequest.push_back(diagRequestInfo);
        }
        
    }

    ifs.close();
    return 0;
}

int32_t
UdsResqestFuncOnIP::StartTestUdsOnIP()
{
    std::cout << "##uds on ip" << std::endl;
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

    if (doipRequestInfo_.size() <= 0) {
        DEBUG_LOG << "error: doipRequestInfo_->requestInfo.size()=0.";
        if (socketApi_->GetLinkStatus()) {
            DEBUG_LOG << "Ipv4TcpClose";
            socketApi_->Ipv4TcpClose();
        }
        return 0;
    }
    for (auto doipInfo : doipRequestInfo_) {
        if (!doipInfo.ip.empty()) {
            ip_ = doipInfo.ip;
            port_ = doipInfo.port;
            ifName_ = doipInfo.ifname;
            break;
        }
    }
    
    for (auto doipInfo : doipRequestInfo_) {
        
        succ = true;
        if (!socketApi_->GetLinkStatus()) {
            int32_t ret = socketApi_->Ipv4TcpConnect(ip_.c_str(), port_, ifName_.c_str());
            if (ret < 0) {
                return -1;
            }
            active_status_ = false;
            seed_ = 0;
        }
        for (auto requestInfo : doipInfo.vecRequest) {
            int32_t ret = DoipRequestUDSdata(requestInfo, doipInfo.ignoreKey);
            if (ret < 0) {
                succ = false;
                break;
            }
        }
        if (succ) {
            PASS_LOG << doipInfo.describe;
        }
        else {
            FAIL_LOG << doipInfo.describe;
            if(!doipInfo.ignoreFail) {
                break;
            }
        }
    }
    if (socketApi_->GetLinkStatus()) {
        DEBUG_LOG << "Ipv4TcpClose";
        socketApi_->Ipv4TcpClose();
    }

    if (!succ) {
        return -1;
    }

    return 0;
}




#if 0
g++ uds_request_.cpp ../../Common/socketraw/socket.cpp ~/git_hozon/netaos/middleware/diag/diag_sa/src/security_algorithm/security_algorithm.cpp \
-I../../../../netaos_thirdparty/x86/jsoncpp/include \
-I../../../../middleware \
-I../../Common -ljsoncpp \
-L../../../../netaos_thirdparty/x86/jsoncpp/lib
int main() {
    UdsResqestFuncOnIP udsRequest;
    udsRequest.StartTestUdsOnIP();
    return 0;
}
#endif


