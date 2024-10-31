/*
 * Copyright (c) Hozon SOC Co., Ltd. 2023-2023. All rights reserved.
 *
 * Description: diag client socket
 */
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <linux/rtnetlink.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <net/if.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <thread>

#include "json/json.h"
#include "common.h"
#include "doip_test_thread.h"

int32_t
DoipResqestSocket::Ipv4TcpDoipPayloadSend(doip_payload_t *doip_payload, bool haveAddr, bool haveRand, int32_t protocol)
{
    int32_t index = 0;
    uint32_t doip_len = 0;
    uint16_t addr = 0;

    doip_len += 2;
    doip_len += 2;
    doip_len += 4;
    if (haveAddr) {
        doip_len += 4;
    }
    if (haveRand && doip_payload->payload_len >= doip_payload->data_length) {
        doip_len += doip_payload->payload_len;
    }
    else {
        doip_len += doip_payload->data_length;
    }
    
    uint8_t *send_data = new uint8_t[doip_len];
    send_data[index++] = doip_payload->version;
    send_data[index++] = doip_payload->ver_inverse;
    addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->payload_type);
    memcpy(send_data + index, &addr, 2);
    index += 2;
    doip_len = LITTLE_TO_BIG_ENDIAN_32(doip_payload->payload_len);
    memcpy(send_data + index, &doip_len, 4);
    index += 4;
    if (haveAddr) {
        addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->source_addr);
        memcpy(send_data + index, &addr, 2);
        index += 2;
        addr = LITTLE_TO_BIG_ENDIAN_16(doip_payload->target_addr);
        memcpy(send_data + index, &addr, 2);
        index += 2;
    }
    if (haveRand && doip_payload->payload_len >= doip_payload->data_length) {
        memcpy(send_data + index, doip_payload->data, doip_payload->data_length);
        index += doip_payload->data_length;
        if (haveAddr) {
            memset(send_data + index, 0, doip_payload->payload_len - doip_payload->data_length - 4);
            index += (doip_payload->payload_len - doip_payload->data_length - 4);
        }
        else {
            memset(send_data + index, 0, doip_payload->payload_len - doip_payload->data_length);
            index += (doip_payload->payload_len - doip_payload->data_length);
        }
    }
    else {
        memcpy(send_data + index, doip_payload->data, doip_payload->data_length);
        index += doip_payload->data_length;
    }

    printfVecHex("send:", send_data, index);
    if (protocol == 0) {
        socketApi_->Ipv4TcpSendData(send_data, index);
    }
    else {
        socketApiUdp_->Ipv4UdpSendData(send_data, index);
    }
    delete[] send_data;

    return index;
}

int32_t
DoipResqestSocket::Ipv4TcpDoipPayloadRecv(doip_payload_t *response, uint32_t timeoutMs, int32_t protocol)
{
    int ret;
    uint8_t recv_data[4096];

    memset(recv_data, 0, sizeof(recv_data));
    if (protocol == 0) {
        ret = socketApi_->Ipv4TcpRecvData(recv_data, 8, timeoutMs);
        if (ret != 8) {
            FAIL_LOG << "recv tcp data error, ret " << ret;
            return -1;
        }
    }
    else {
        ret = socketApiUdp_->Ipv4UdpRecvData(recv_data, sizeof(recv_data), timeoutMs);
        if(ret < 0) {
            FAIL_LOG << "recv udp data error, ret " << ret;
            return -1;
        }
    }

    int32_t index = 0;
    response->version = recv_data[index++];
    response->ver_inverse = recv_data[index++];
    response->payload_type = (recv_data[index] << 8) | recv_data[index + 1];
    index += 2;
    response->payload_len = (recv_data[index] << 24) | (recv_data[index + 1] << 16) | (recv_data[index + 2] << 8) | recv_data[index + 3];
    index += 4;

    if (response->payload_len <= 0) {
        //DEBUG_LOG << "response payload date len is 0";
        printfVecHex("recv:", recv_data, index);
        return index;
    }

    if (sizeof(recv_data) < response->payload_len + index) {
        socketApi_->Ipv4TcpClose();
        FAIL_LOG << "sizeof recv_buf error, payload_len " << response->payload_len + index;
        return -1;
    }
    if (protocol == 0) {
        ret = socketApi_->Ipv4TcpRecvData(recv_data + index, response->payload_len, timeoutMs);
    }
    else {
        ret = (int)response->payload_len;// UDP have recv end;
    }
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
DoipResqestSocket::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
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

bool
DoipResqestSocket::CheckResponse(TestItem::DoipResponse *jsonResp, doip_payload_t *response, bool haveAddr)
{
    doip_payload_t doip_payload = {0};
    doip_payload.version = std::stoi(jsonResp->version, 0, 16);
    doip_payload.ver_inverse = std::stoi(jsonResp->ver_inverse, 0, 16);
    doip_payload.payload_type = std::stoi(jsonResp->payload_type, 0, 16);
    doip_payload.payload_len = jsonResp->payload_len;
    if (haveAddr) {
        doip_payload.source_addr = std::stoi(jsonResp->source_addr, 0, 16);
        doip_payload.target_addr = std::stoi(jsonResp->target_addr, 0, 16);
    }
    
    jsonResp->payload_data.erase(std::remove_if(jsonResp->payload_data.begin(),\
        jsonResp->payload_data.end(), ::isspace),\
        jsonResp->payload_data.end() );
    std::vector<uint8_t > vec_jsonResp;
    ParseFromBcd(jsonResp->payload_data, vec_jsonResp);
    jsonResp->isExactMatch = true;
    if (!jsonResp->payload_data.empty() && jsonResp->payload_data.back() == '.') {
        jsonResp->isExactMatch = false;
    }

    if (doip_payload.version != response->version) {
        FAIL_LOG << "version error " << response->version;
        return false;
    }
    if (doip_payload.ver_inverse != response->ver_inverse) {
        FAIL_LOG << "ver_inverse error " << response->ver_inverse;
        return false;
    }
    if (doip_payload.payload_type != response->payload_type) {
        FAIL_LOG << "payload_type error " << response->payload_type;
        return false;
    }
    if (jsonResp->isExactMatch && doip_payload.payload_len != response->payload_len) {
        FAIL_LOG << "payload_len error " << response->payload_len;
        return false;
    }
    if (haveAddr && response->payload_len >=4 && doip_payload.source_addr != response->source_addr) {
        FAIL_LOG << "source_addr error " << response->source_addr;
        return false;
    }
    if (haveAddr && response->payload_len >=4 && doip_payload.target_addr != response->target_addr) {
        FAIL_LOG << "target_addr error " << response->target_addr;
        return false;
    }

    int32_t payloadLen = response->payload_len < 4 ? (response->payload_len) : (response->payload_len - 4);
    std::vector<uint8_t > vec_response(response->data, response->data + payloadLen);
    if (jsonResp->isExactMatch && vec_response != vec_jsonResp) {
        FAIL_LOG << "payload data error ";
        return false;
    }

    return true;
}

int32_t
DoipResqestSocket::ParseJsonFile(const std::string& filename, TestInfo& info)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return -1;
    }

    // 解析JSON
    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        FAIL_LOG << "parseJson error! " << errs;
        return -1;
    }
    ifs.close();
    
    // 解析ip地址和端口号
    if (info.ip_addr.size() == 0) {
        info.ip_addr = root["ip_addr"].asString();
        info.port = root["port"].asInt();
        info.ifname = root["ifname"].asString();
    }

    // 解析测试项
    Json::Value testItems = root["test"];
    for (Json::Value::ArrayIndex i = 0; i < testItems.size(); ++i) {
        TestItem item;
        item.protocol_type = testItems[i]["protocol_type"].asString();
        item.timeout = testItems[i].get("timeout", 5000).asInt();
        item.retryCounts = testItems[i].get("retryCounts", 1).asInt();
        item.tcp_close = testItems[i].get("tcp_close", false).asBool();
        item.describe = testItems[i]["describe"].asString();
        item.delay_timems = testItems[i].get("delay_timems", 0).asInt();
        item.ignoreFail = testItems[i].get("ignoreFail", false).asBool();

        // 解析doip_request
        Json::Value doipReq = testItems[i]["doip_request"];
        for (Json::Value::ArrayIndex j = 0; j < doipReq.size(); ++j) {
            TestItem::DoipRequest req;
            req.version = doipReq[j]["version"].asString();
            req.ver_inverse = doipReq[j]["ver_inverse"].asString();
            req.payload_type = doipReq[j]["payload_type"].asString();
            req.payload_len = doipReq[j]["payload_len"].asInt();
            req.source_addr = doipReq[j]["source_addr"].asString();
            req.target_addr = doipReq[j]["target_addr"].asString();
            req.payload_data = doipReq[j]["payload_data"].asString();
            item.doip_request.push_back(req);
        }

        // 解析doip_response
        doipReq = testItems[i]["doip_response"];
        for (Json::Value::ArrayIndex j = 0; j < doipReq.size(); ++j) {
            TestItem::DoipResponse resp;
            resp.version = doipReq[j]["version"].asString();
            resp.ver_inverse = doipReq[j]["ver_inverse"].asString();
            resp.payload_type = doipReq[j]["payload_type"].asString();
            resp.payload_len = doipReq[j]["payload_len"].asInt();
            resp.source_addr = doipReq[j]["source_addr"].asString();
            resp.target_addr = doipReq[j]["target_addr"].asString();
            resp.payload_data = doipReq[j]["payload_data"].asString();
            item.doip_response.push_back(resp);
        }

        info.test_items.push_back(item);
    }

    return 0;
}

int32_t
DoipResqestSocket::TestDoipThread(TestInfo info, int32_t threadId)
{
    DEBUG_LOG << "thread start " << threadId;

    bool succ = true;
    ip_ = info.ip_addr;
    port_ = info.port;
    ifName_ = info.ifname;
    int32_t ret = socketApi_->Ipv4TcpConnect(ip_.c_str(), port_, ifName_.c_str());
    if (ret < 0) {
        result = -1;
        return -1;
    }

    ret = socketApiUdp_->CreateSocket(AF_INET, SOCK_DGRAM, 0);
    if (ret < 0) {
        result = -1;
        return -1;
    }
    ret = socketApiUdp_->Ipv4UdpBindAddr(ip_.c_str(), port_);
    if (ret < 0) {
        result = -1;
        return -1;
    }

    for(auto test_items: info.test_items) {
        
        int32_t protocol = 0;
        if (test_items.protocol_type == "TCP") {
            protocol = 0;
        }
        else {
            protocol = 1;
        }
        succ = true;
        
        if (!socketApi_->GetLinkStatus() && protocol == 0) {
            ret = socketApi_->Ipv4TcpConnect(ip_.c_str(), port_, ifName_.c_str());
            if (ret < 0) {
                result = -1;
                return -1;
            }
        }
        for(uint32_t i = 0; i < test_items.doip_request.size(); i++) {
            doip_payload_t doip_payload;
            bool haveRand = false;//测试超大数据，request设置类似"10 01 ..."，依据payload_length填充
            bool haveAddr = true;
            memset(&doip_payload, 0, sizeof(doip_payload));
            doip_payload.version = std::stoi(test_items.doip_request[i].version, 0, 16);
            doip_payload.ver_inverse = std::stoi(test_items.doip_request[i].ver_inverse, 0, 16);
            doip_payload.payload_type = std::stoi(test_items.doip_request[i].payload_type, 0, 16);
            doip_payload.payload_len = test_items.doip_request[i].payload_len;
            if (test_items.doip_request[i].source_addr.size() > 0 && test_items.doip_request[i].target_addr.size() > 0) {
                doip_payload.source_addr = std::stoi(test_items.doip_request[i].source_addr, 0, 16);
                doip_payload.target_addr = std::stoi(test_items.doip_request[i].target_addr, 0, 16);
            }
            else {
                haveAddr = false;
            }
            
            test_items.doip_request[i].payload_data.erase(std::remove_if(test_items.doip_request[i].payload_data.begin(), \
                test_items.doip_request[i].payload_data.end(), 
                [](char c) {
                    return (c == ' ' || c == '\n' || c == '\r' ||
                            c == '\t' || c == '\v' || c == '\f');
                }),
                test_items.doip_request[i].payload_data.end());
            if (!test_items.doip_request[i].payload_data.empty() && test_items.doip_request[i].payload_data.back() == '.') {
                haveRand = true;
            }
            std::vector<uint8_t > vec_request;
            ParseFromBcd(test_items.doip_request[i].payload_data, vec_request);
            doip_payload.data = NULL;
            if (vec_request.size() > 0) {
                doip_payload.data = new uint8_t[vec_request.size()]();
                memcpy(doip_payload.data, &(vec_request[0]), vec_request.size());
                doip_payload.data_length = vec_request.size();
            }
            ret = Ipv4TcpDoipPayloadSend(&doip_payload, haveAddr, haveRand, protocol);
            if (ret < 0) {
                if (vec_request.size() > 0) {
                    delete[] doip_payload.data;
                }
                succ = false;
                break;
            }
            if (vec_request.size() > 0) {
                delete[] doip_payload.data;
            }
        }
        for(uint32_t i = 0; i < test_items.doip_response.size(); i++) {
            doip_payload_t doip_payload;
            bool haveAddrResp = true;
            doip_payload.data = NULL;
            ret = Ipv4TcpDoipPayloadRecv(&doip_payload, test_items.timeout, protocol);
            if (ret < 0) {
                //delete[] doip_payload.data;
                succ = false;
                break;
            }
            if (test_items.doip_response[i].source_addr.size() <= 0 || test_items.doip_response[i].target_addr.size() <= 0) {
                haveAddrResp = false;
            }
            ret = CheckResponse(&test_items.doip_response[i], &doip_payload, haveAddrResp);
            if (ret) {
                DEBUG_LOG << "response succ";
            }
            else {
                DEBUG_LOG << "response fail";
                if (doip_payload.payload_len > 0) {
                    delete[] doip_payload.data;
                }
                succ = false;
                break;
            }
            if (doip_payload.payload_len > 0) {
                delete[] doip_payload.data;
            }
        }
        if (test_items.sever_close_wait > 0) {
            DEBUG_LOG << "wait sever_close_wait " << test_items.sever_close_wait;
            uint8_t recv_data[4096];
            if (protocol == 0) {
                ret = socketApi_->Ipv4TcpRecvData(recv_data, 8, test_items.sever_close_wait);
                if (ret != -2) {
                    FAIL_LOG << "recv tcp data error, ret " << ret;
                    result = -1;
                    return -1;
                }
                socketApi_->Ipv4TcpClose();
            }
            else {
                ret = socketApiUdp_->Ipv4UdpRecvData(recv_data, sizeof(recv_data), test_items.sever_close_wait);
                if(ret != -2) {
                    FAIL_LOG << "recv udp data error, ret " << ret;
                    result = -1;
                    return -1;
                }
                //socketApiUdp_->Ipv4UdpClose();
            }
        }

        if (test_items.tcp_close) {
            DEBUG_LOG << "Ipv4TcpClose";
            socketApi_->Ipv4TcpClose();
        }

        if (succ) {
            PASS_LOG << test_items.describe;
            INFO_LOG << "";
        }
        else {
            FAIL_LOG << test_items.describe;
            INFO_LOG << "";
            if(!test_items.ignoreFail) {
                break;
            }
        }
        if (test_items.delay_timems > 0) {
            DEBUG_LOG << "delay_timems " << test_items.delay_timems;
            usleep(test_items.delay_timems*1000);
        }
    }
    DEBUG_LOG << "Test all file end";
    if (socketApi_->GetLinkStatus()) {
        DEBUG_LOG << "Ipv4TcpClose";
        socketApi_->Ipv4TcpClose();
    }
    socketApiUdp_->Ipv4UdpClose();
    
    if (!succ) {
        result = -1;
        return -1;
    }

    result = 0;
    return 0;
}

#if 0
int main() {
    DoipResqestSocket requestSocket;
    requestSocket.StartTestDoip();
    return 0;
}

#endif
