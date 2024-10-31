/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: fault manager client sample
*/
// someip
#include <ara/core/initialization.h>
#include "hozon/netaos/v1/mcufaultservice_proxy.h"

#include "log/include/logging.h"
#include "phm/include/phm_client.h"
#include <signal.h>
#include <string.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <thread>
#include <memory>
#include <unordered_map>

using namespace hozon::netaos::phm;
std::shared_ptr<PHMClient> phm_client_ptr = nullptr;
bool isAvailable = false;
uint8_t stopFlag = 0;
std::shared_ptr<hozon::netaos::v1::proxy::McuFaultServiceProxy> proxy_;
bool SomeipInitFlag = false;

std::vector<uint64_t> Fault_Info =
{
    1001700, 1001702, 1001704, 1600706
};

std::vector<std::string> Fault_Detail_Info =
{
    "故障来源: 平台故障清单 故障名称[大分类]: 视觉处理硬件严重故障 故障名称: 前视30视觉处理硬件严重故障",
};

const std::unordered_map<uint32_t, std::string> map_dtc_fault_info =
{
    {5701704, "800401    0x570048    合众SoC故障 [进程保活Alive监控异常: 前雷达进程保活异常]"},
    {5701960, "800402    0x570148    合众SoC故障 [进程保活Alive监控异常: 角雷达进程保活异常]"},
    {5824711, "1601107   0x58E0C7    合众定位故障 [各模块间异常: vio位姿跳变]"}
};

void SigHandler(int signum)
{
    std::cout << "---  SigHandler enter, signum [" << signum << "] ---" << std::endl;
    stopFlag = 1;
    if (phm_client_ptr) {
        phm_client_ptr->Deinit();
        phm_client_ptr = nullptr;
    }

    exit(0);
}

void
serviceAvailabilityCallback(ara::com::ServiceHandleContainer<ara::com::HandleType> handles)
{
#ifdef BUILD_FOR_ORIN
    // std::cout << "Fm tool serviceAvailabilityCallback";
    if (handles.size() > 0U) {
        if (proxy_ == nullptr) {
            std::cout << "Fm tool serviceAvailabilityCallback created proxy\n";
            proxy_ = std::make_shared<hozon::netaos::v1::proxy::McuFaultServiceProxy>( handles[ 0 ] );
            if ( proxy_ == nullptr ) {
                // std::cout << "Fm tool serviceAvailabilityCallback create proxy failed";
            }
        }
    }
    else {
        // std::cout << "Fm tool serviceAvailabilityCallback service disconnected\n";
        proxy_ = nullptr;
    }
#endif
}

void SomeipInit()
{
#ifdef BUILD_FOR_ORIN
    ara::core::Initialize();

    hozon::netaos::v1::proxy::McuFaultServiceProxy::StartFindService(
        []( ara::com::ServiceHandleContainer<ara::com::HandleType> handles, ara::com::FindServiceHandle handler ) {
            (void) handler;
            // std::cout << "Fm tool Init StartFindService size:" << handles.size();
            serviceAvailabilityCallback( std::move( handles ) );
        },
        ara::com::InstanceIdentifier("1")
    );
#endif
}

void ServiceAvailableCallback(bool bResult)
{
    // std::cout << "PhmTest ServiceAvailableCallback bResult: " << bResult << std::endl;
    isAvailable = bResult;
}

void FaultReceiveCallback(const ReceiveFault_t& fault)
{
    std::cout << "PhmTest FaultReceiveCallback faultId: " << fault.faultId
              << " faultObj: " << static_cast<uint>(fault.faultObj)
              << " faultStatus: " << fault.faultStatus
              << " faultOccurTime: " << fault.faultOccurTime
              << " faultDomain: " << fault.faultDomain
              << " faultDes: " << fault.faultDes << std::endl;

    for (auto& item : fault.faultCluster) {
        std::cout << "PhmTest FaultReceiveCallback clusterName: " << item.clusterName
                  << " clusterValue: " << (int)item.clusterValue << std::endl;
    }
}

std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = " ")
{
    std::regex re(regexStr);
    std::sregex_token_iterator first{inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

std::vector<std::string> GetParam()
{
    std::string s;
    do {
        getline(std::cin, s);
    }
    while (s.empty());
    std::vector<std::string> vs = Split(s);
    return vs;
}

void QueryDtcInfo(uint32_t dtcCode, std::string& str)
{
    auto itr = map_dtc_fault_info.find(dtcCode);
    if (itr != map_dtc_fault_info.end()) {
        str = itr->second;
    }
    else {
        str = "没有查到该DTC的信息!";
    }
}

void CollectionFileCb(std::vector<std::string>& vAllFile)
{
    std::cout << "\nfm tool CollectionFileCb size:" << vAllFile.size() << "\n";
    for (auto f : vAllFile) {
        printf("CollectionFileCb file: %s\n", f.data());
    }

    return;
}

int main(int argc, char* argv[])
{
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);
    phm_client_ptr.reset(new PHMClient());
    phm_client_ptr->Init("", ServiceAvailableCallback, FaultReceiveCallback);
    // check client available

    int inputNum = 0;
    while (1) {
        // content
        std::cout << "FM 故障注入工具\n" \
                    "1: 故障注入测试\n" \
                    "2: 立即更新故障分析文件\n" \
                    "3: 查询DTC信息\n" \
                    "4: MCU故障注入测试\n" \
                    "88: 退出\n" \
                    "b: 返回上一级\n" \
                    "input:" ;

        // get input number
        if (scanf("%d", &inputNum)) {}
        int iTmp = 0;
        do {
            iTmp = getchar();
        } while ((iTmp != '\n') && (iTmp != EOF));

        switch (inputNum) {
        case 1:
            {
                for(;;) {
                    std::cout << "故障注入格式: 故障Id 故障Obj 上报类型(产生[1]恢复[0]) \ninput:";
                    std::vector<std::string> vInput = GetParam();
                    if (vInput.size() == 1 && vInput[0] == "b") {
                        break;
                    }

                    if (vInput.size() < 3) {
                        std::cout << "参数输入异常, 请重新输入参数" << std::endl;
                        continue;
                    }

                    if (0 == std::isdigit(vInput[0].at(0))
                        || 0 == isdigit(vInput[1].at(0))
                        || 0 == isdigit(vInput[2].at(0))) {
                        std::cout << "参数输入异常, 请重新输入参数" << std::endl;
                        continue;
                    }

                    int faultId = std::stoi(vInput[0]);
                    int faultObj = std::stoi(vInput[1]);
                    int status = std::stoi(vInput[2]);
                    SendFault_t sendFault(faultId, faultObj, status);
                    phm_client_ptr->ReportFault(sendFault);
                }
            }
            break;
        case 2:
            {
                phm_client_ptr->GetDataCollectionFile(CollectionFileCb);
            }
            break;
        case 3:
            {
                std::cout << "单个or多个DTC查询 例: 5701704 5701960...\ninput:";
                std::vector<std::string> vInput = GetParam();
                if (vInput.size() == 1 && vInput[0] == "b") {
                    break;
                }

                std::cout << "DTC(DEC)    Fault     DTC(HEX)    故障来源    [故障大分类: 故障名称]" << std::endl;
                for (auto& strDtc : vInput) {
                    if (0 == std::isdigit(strDtc.at(0))) {
                        continue;
                    }

                    int iDtc = stoi(strDtc);
                    // printf("strDtc:%s,iDtc:%d\n", strDtc.data(), iDtc);
                    std::string dtcInfo;
                    QueryDtcInfo(static_cast<uint32_t>(iDtc), dtcInfo);
                    std::cout << std::setw(8) << std::setfill(' ') << static_cast<uint32_t>(iDtc) << "    " << dtcInfo.c_str() << std::endl;
                }
                std::cout<< std::endl;
            }
            break;
        case 4:
            {
                #ifdef BUILD_FOR_ORIN
                if (!SomeipInitFlag) {
                    SomeipInit();
                    SomeipInitFlag = true;
                }

                for(;;) {
                    std::cout << "\n故障注入格式: 故障Id 故障Obj 上报类型(产生[1]恢复[0]) \ninput:";
                    std::vector<std::string> vInput = GetParam();
                    if (vInput.size() == 1 && vInput[0] == "b") {
                        break;
                    }

                    if (vInput.size() < 3) {
                        std::cout << "参数输入异常, 请重新输入参数" << std::endl;
                        continue;
                    }

                    if (0 == std::isdigit(vInput[0].at(0))
                        || 0 == isdigit(vInput[1].at(0))
                        || 0 == isdigit(vInput[2].at(0))) {
                        std::cout << "参数输入异常, 请重新输入参数" << std::endl;
                        continue;
                    }

                    int faultId = std::stoi(vInput[0]);
                    int faultObj = std::stoi(vInput[1]);
                    int status = std::stoi(vInput[2]);

                    hozon::netaos::FaultDataStruct FaultData;
                    FaultData.faultId = faultId;
                    FaultData.faultObj = faultObj;
                    FaultData.faultStatus = status;
                    for (size_t i = 0; i < 60; ++i) {
                        FaultData.postProcessArray[i] = 0;
                    }

                    if (proxy_ == nullptr) {
                        std::cout << "McuFaultServiceProxy is nullptr" << std::endl;
                        continue;
                    }

                    auto FaultReportResult = proxy_->FaultReport(FaultData);
                    if (ara::core::future_status::timeout == FaultReportResult.wait_for(std::chrono::milliseconds(50))) {
                        std::cout << "Fm tool FaultReportToMCU report fault future timeout\n";
                        continue;
                    }

                    auto Future = FaultReportResult.GetResult();
                    if (Future.HasValue()) {
                        auto output = Future.Value();
                        std::cout << "Fm tool FaultReportToMCU result:" << static_cast<int>(output.FaultReportResult) << "\n";
                    }
                    else {
                        std::cout << "Fm tool FaultReportToMCU not has value\n";
                    }
                }

                #endif
            }
            break;
        case 88:
            phm_client_ptr->Deinit();
            return 0;
        default:
            std::cout << "请正确输入参数...\n\n";
            break;
        }
    }

    return 0;
}
