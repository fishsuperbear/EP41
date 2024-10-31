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
#include <memory>

#include "json/json.h"
#include "common.h"
#include "doip_request_test.h"
#include "doip_test_thread.h"

int32_t
DoipRequestTest::ParseJsonFile(const std::string& filename, TestInfo& info)
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
        item.describe = testItems[i]["describe"].asString();
        item.delay_timems = testItems[i].get("delay_timems", 0).asInt();
        item.ignoreFail = testItems[i].get("ignoreFail", false).asBool();
        item.tcp_close = testItems[i].get("tcp_close", false).asBool();
        item.sever_close_wait = testItems[i].get("sever_close_wait", 0).asUInt();

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
DoipRequestTest::ParseJsonFile2(const std::string& filename, std::vector<TestInfo>& info)
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
    

    for (const auto& obj : root) {
        TestInfo testObj;

        // 解析ip地址和端口号
        testObj.ip_addr = obj["ip_addr"].asString();
        testObj.port = obj["port"].asInt();
        testObj.ifname = obj["ifname"].asString();
        testObj.thread_delay = obj["thread_start_delay"].asInt();

        // 解析测试项
        Json::Value testItems = obj["test"];
        for (Json::Value::ArrayIndex i = 0; i < testItems.size(); ++i) {
            TestItem item;
            item.protocol_type = testItems[i]["protocol_type"].asString();
            item.timeout = testItems[i].get("timeout", 5000).asInt();
            item.retryCounts = testItems[i].get("retryCounts", 1).asInt();
            item.describe = testItems[i]["describe"].asString();
            item.delay_timems = testItems[i].get("delay_timems", 0).asInt();
            item.ignoreFail = testItems[i].get("ignoreFail", false).asBool();
            item.tcp_close = testItems[i].get("tcp_close", false).asBool();
            item.sever_close_wait = testItems[i].get("sever_close_wait", 0).asUInt();

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

            testObj.test_items.push_back(item);
        }
        
        info.push_back(testObj);
    }

    return 0;
}
int32_t
DoipRequestTest::StartTestDoip()
{
    std::cout << "##Test doip" << std::endl;

    int32_t ret = 0;
    std::vector<std::string> json_files;
    DIR* dir = opendir("../Conf/doip");

    if (dir) {
        dirent* entry;
        while ((entry = readdir(dir))) {
            std::string filename = entry->d_name;
            if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json") {
                json_files.push_back(filename);
            }
        }
        closedir(dir);
    }
    std::sort(json_files.begin(), json_files.end());
    for(auto file: json_files) {
        DEBUG_LOG << "JsonFile :" << file;
    }

    for(auto file: json_files) {
        if (file.find("task") == file.npos) {
            ParseJsonFile("../Conf/doip/" + file, doipRequestInfo_);
        }
        else {
            std::vector<TestInfo> testinfo;
            ParseJsonFile2("../Conf/doip/" + file, testinfo);
            doipRequestInfo2_.push_back(testinfo);
        }
    }
    
    // for (auto doipInfo : doipRequestInfo_) {
    //     if (!doipInfo.ip_addr.empty()) {
    //         ip_ = doipRequestInfo_[0].ip_addr;
    //         port_ = doipRequestInfo_[0].port;
    //         ifName_ = doipRequestInfo_[0].ifname;
    //         break;
    //     }
    // }

    //获取vin和eid，每辆车的vin不一样，没办法做到统一配置，使用动态获取
    SocketApiUdp *socketApiUdp = new SocketApiUdp();
    ret = socketApiUdp->CreateSocket(AF_INET, SOCK_DGRAM, 0);
    if (ret < 0) {
        FAIL_LOG << "CreateSocket error, ret " << ret;
        return -1;
    }
    ret = socketApiUdp->Ipv4UdpBindAddr(doipRequestInfo_.ip_addr.c_str(), doipRequestInfo_.port);
    if (ret < 0) {
        FAIL_LOG << "Ipv4UdpBindAddr error, ret " << ret;
        return -1;
    }
    uint8_t buffer[100] = {0x02,0xfd,0x00,0x01,0x00,0x00,0x00,0x00};
    printfVecHex("send:", buffer, 8);
    socketApiUdp->Ipv4UdpSendData(buffer, 8);
    if (ret < 0) {
        FAIL_LOG << "send udp data error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    memset(buffer, 0, sizeof(buffer));
    ret = socketApiUdp->Ipv4UdpRecvData(buffer, sizeof(buffer), 2000);
    if(ret < 0) {
        FAIL_LOG << "recv udp data error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    printfVecHex("recv:", buffer, ret);
    if(buffer[0] != 0x02 || buffer[1] != 0xfd
        || buffer[2] != 0x00 || buffer[3] != 0x04) {
        FAIL_LOG << "recv udp data(0004) error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    PASS_LOG << "Vehicle announcement message succ.";
    uint8_t buff_eid[100] = {0x02,0xfd,0x00,0x02,0x00,0x00,0x00,0x06};
    memcpy(buff_eid + 8, buffer + 8 + 17 + 2, 6);
    printfVecHex("send:", buff_eid, 14);
    socketApiUdp->Ipv4UdpSendData(buff_eid, 14);
    if(ret < 0) {
        FAIL_LOG << "send udp data with eid error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    memset(buff_eid, 0, sizeof(buff_eid));
    ret = socketApiUdp->Ipv4UdpRecvData(buff_eid, sizeof(buff_eid), 2000);
    if(ret < 0) {
        FAIL_LOG << "recv udp data eid error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    printfVecHex("recv:", buff_eid, ret);
    if(buff_eid[0] != 0x02 || buff_eid[1] != 0xfd
        || buff_eid[2] != 0x00 || buff_eid[3] != 0x04) {
        FAIL_LOG << "recv udp data with eid error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    PASS_LOG << "Vehicle announcement message with eid succ.";
    uint8_t buff_vin[100] = {0x02,0xfd,0x00,0x03,0x00,0x00,0x00,0x11};
    memcpy(buff_vin + 8, buffer + 8, 17);
    printfVecHex("send:", buff_vin, 25);
    socketApiUdp->Ipv4UdpSendData(buff_vin, 25);
    if(ret < 0) {
        FAIL_LOG << "recv udp data with vin error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    memset(buff_vin, 0, sizeof(buff_vin));
    ret = socketApiUdp->Ipv4UdpRecvData(buff_vin, sizeof(buff_vin), 2000);
    if(ret < 0) {
        FAIL_LOG << "recv udp data vin error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    printfVecHex("recv:", buff_vin, ret);
    if(buff_vin[0] != 0x02 || buff_vin[1] != 0xfd
        || buff_vin[2] != 0x00 || buff_vin[3] != 0x04) {
        FAIL_LOG << "recv udp data with vin error, ret " << ret;
        socketApiUdp->Ipv4UdpClose();
        return -1;
    }
    PASS_LOG << "Vehicle announcement message with vin succ.";
    socketApiUdp->Ipv4UdpClose();
    delete socketApiUdp;



    //单client端的测试
    if (doipRequestInfo_.test_items.size() > 0) {
        DoipResqestSocket doip_request0;
        ret = doip_request0.TestDoipThread(doipRequestInfo_, 0xffff);
        if (ret < 0) {
            return -1;
        }
    }

    //多client端的测试，使用多线程的方式
    for (auto test_items: doipRequestInfo2_) {
        DEBUG_LOG <<"=====================================================task test start";
        int32_t threadId = 0;
        std::thread threads[test_items.size()];
        DoipResqestSocket doip_request[test_items.size()];

        for (auto test_items2: test_items) {
            if (test_items2.thread_delay > 0) {
                DEBUG_LOG << "thread_id: " << threadId << ", thread_delay " << test_items2.thread_delay;
                std::this_thread::sleep_for(std::chrono::milliseconds(test_items2.thread_delay));
            }
            DEBUG_LOG << "thread_id: " << threadId << ", create";
            threads[threadId] = std::thread(&DoipResqestSocket::TestDoipThread, &doip_request[threadId], test_items2, threadId);
            threadId++;
        }
        for (int32_t i = 0; i < threadId; i++) {
            threads[i].join();
        }
        for (int32_t i = 0; i < threadId; i++) {
            if (doip_request[i].result < 0) {
                return -1;
            }
        }
        DEBUG_LOG <<"=====================================================task test end";
    }


    return 0;
}


#if 0
int main() {
    DoipRequestTest requestSocket;
    requestSocket.StartTestDoip();
    return 0;
}

#endif
