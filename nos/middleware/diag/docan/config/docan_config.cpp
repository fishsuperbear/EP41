/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanConfig implement
 */

#include "docan_config.h"
#include <fstream>
#include "json/json.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

#define DOCAN_CONFIG_FILE_FOR_MDC   ("/opt/usr/diag_update/mdc-llvm/conf/docan_config.json")
#define DOCAN_CONFIG_FILE_FOR_J5    ("/userdata/diag_update/j5/conf/docan_config.json")
#define DOCAN_CONFIG_FILE_ORIN      ("/app/runtime_service/diag_server/conf/docan_config.json")
#define DOCAN_CONFIG_FILE_DEFAULT   ("/app/runtime_service/diag_server/conf/docan_config.json")

DocanConfig *DocanConfig::instancePtr_ = nullptr;
std::mutex DocanConfig::instance_mtx_;

DocanConfig*
DocanConfig::instance()
{
    if (nullptr == instancePtr_)
    {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_)
        {
            instancePtr_ = new DocanConfig();
        }
    }
    return instancePtr_;
}

void
DocanConfig::destroy()
{
    if (nullptr != instancePtr_) {
        delete instancePtr_;
        instancePtr_ = nullptr;
    }
}

DocanConfig::DocanConfig()
{
}

DocanConfig::~DocanConfig()
{
}

int32_t DocanConfig::Init(void)
{
    DOCAN_LOG_D("Init()");
    loadConfig();
    return 0;
}

int32_t DocanConfig::Start(void)
{

    return 0;
}

int32_t DocanConfig::Stop(void)
{
    return 0;
}

int32_t DocanConfig::Deinit(void)
{
    return 0;
}

int32_t
DocanConfig::loadConfig()
{
    DOCAN_LOG_D("loadConfig");
    // read docan config json file
    int32_t ret = -1;

    std::string configFile;

#ifdef BUILD_FOR_MDC
    configFile = DOCAN_CONFIG_FILE_FOR_MDC;
#elif BUILD_FOR_J5
    configFile = DOCAN_CONFIG_FILE_FOR_J5;
#elif BUILD_FOR_ORIN
    configFile = DOCAN_CONFIG_FILE_ORIN;
#else
    configFile = DOCAN_CONFIG_FILE_DEFAULT;
#endif

    if (0 != access(configFile.c_str(), F_OK)) {
        DOCAN_LOG_E("Config file: %s, is not existed!", configFile.c_str());
        return ret;
    }

    Json::Value rootReder;
    Json::CharReaderBuilder readBuilder;
    std::ifstream ifs(configFile);
    std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
    JSONCPP_STRING errs;
    if (!Json::parseFromStream(readBuilder, ifs, &rootReder, &errs)) {
        DOCAN_LOG_E("parseFromStream failed, errs: %s.", errs.c_str());
        return ret;
    }

    if (rootReder["DocanNodesConfiguration"].size() > 0 && can_ecu_info_list_.size() > 0) {
        can_ecu_info_list_.clear();
    }
    for (uint8_t index = 0; index < rootReder["DocanNodesConfiguration"].size(); ++index) {
        N_EcuInfo_t info;
        Json::Value ecu = rootReder["DocanNodesConfiguration"][index];
        info.ecu_name = ecu["Name"].asString();
        info.if_name = ecu["IfName"].asString();
        info.can_type = (ecu["CanType"].asString() == "canfd") ? 2
                      : (ecu["CanType"].asString() == "Canfd") ? 2
                      : 1;
        info.diag_type = (ecu["DiagType"].asString() == "Remote") ? 2
                       : (ecu["DiagType"].asString() == "remote") ? 2
                       : 1;
        info.ta_type =  (ecu["TaType"].asString() == "Fucntional") ? 2
                       : (ecu["TaType"].asString() == "functional") ? 2
                       : 1;
        info.address_logical = std::stoi(ecu["LogicalAddr"].asString(), 0, 16);
        info.canid_tx = std::stoi(ecu["CanidTx"].asString(), 0, 16);
        info.canid_rx = std::stoi(ecu["CanidRx"].asString(), 0, 16);
        info.address_functional = std::stoi(ecu["FunctionAddr"].asString(), 0, 16);
        info.BS = ecu["BS"].asUInt();
        info.STmin = ecu["STmin"].asUInt();
        info.N_WFTmax = (0 != ecu["WFTmax"].asUInt()) ? ecu["WFTmax"].asUInt() : 5;
        info.N_As = (0 != ecu["As"].asUInt()) ? ecu["As"].asUInt() : 30;
        info.N_Ar = (0 != ecu["Ar"].asUInt()) ? ecu["Ar"].asUInt() : 30;
        info.N_Bs = (0 != ecu["Bs"].asUInt()) ? ecu["Bs"].asUInt() : 90;
        info.N_Br = (0 != ecu["Br"].asUInt()) ? ecu["Br"].asUInt() : 50;
        info.N_Cs = (0 != ecu["Cs"].asUInt()) ? ecu["Cs"].asUInt() : 150;
        info.N_Cr = (0 != ecu["Cr"].asUInt()) ? ecu["Cr"].asUInt() : 50;
        can_filter filter;
        filter.can_id = info.canid_rx;
        filter.can_mask = 0x700;
        info.filters.push_back(filter);
        can_ecu_info_list_.push_back(info);
        DOCAN_LOG_D("ecu: %s, ifName: %s, logcalAddr: %X, canid_tx: %X, canid_rx: %X, funcAddr: %X.",
            info.ecu_name.c_str(), info.if_name.c_str(), info.address_logical, info.canid_tx, info.canid_rx, info.address_functional);
    }

    if (rootReder["DocanRoutingTable"].size() > 0 && can_route_info_list_.size() > 0) {
        can_route_info_list_.clear();
    }
    for (uint8_t index = 0; index < rootReder["DocanRoutingTable"].size(); ++index) {
        N_RouteInfo_t info;
        Json::Value route = rootReder["DocanRoutingTable"][index];
        info.route_name = route["Name"].asString();
        info.if_name = route["IfName"].asString();
        info.address_logical = std::stoi(route["LogicalAddr"].asString(), 0, 16);
        info.address_functional = std::stoi(route["FunctionAddr"].asString(), 0, 16);
        for (uint8_t idx = 0; idx < route["ForwordTable"].size(); ++idx) {
            Json::Value forword = route["ForwordTable"][idx];
            N_ForwordInfo_t forwordInfo;
            forwordInfo.gw_canid_tx = std::stoi(forword["GwCanidTx"].asString(), 0, 16);
            forwordInfo.gw_canid_rx = std::stoi(forword["GwCanidRx"].asString(), 0, 16);
            forwordInfo.forword_logical_addr = std::stoi(forword["ForwordLogicalAddr"].asString(), 0, 16);
            forwordInfo.forword_ecu = forword["ForwordEcu"].asString();
            forwordInfo.forword_if_name = forword["ForwordIfName"].asString();
            forwordInfo.forword_canid_tx = std::stoi(forword["ForwordCanidTx"].asString(), 0, 16);
            forwordInfo.forword_canid_rx = std::stoi(forword["ForwordCanidRx"].asString(), 0, 16);
            info.forward_table.push_back(forwordInfo);
        }
        can_route_info_list_.push_back(info);
        DOCAN_LOG_D("route: %s, ifName: %s, logcalAddr: %X, funcAddr: %X.",
            info.route_name.c_str(), info.if_name.c_str(), info.address_logical, info.address_functional);
    }

    ifs.close();
    ret = can_ecu_info_list_.size() + can_route_info_list_.size();
    DOCAN_LOG_D("loadConfig success, can_ecu_info_list_ size: %ld, can_route_info_list_ size: %ld.", can_ecu_info_list_.size(), can_route_info_list_.size());
    return ret;
}

uint16_t
DocanConfig::getEcu(uint16_t canid_rx)
{
    uint16_t ecu = 0;
    for (auto it : can_ecu_info_list_) {
        if (it.canid_rx == canid_rx) {
            ecu = it.address_logical;
        }
    }
    return ecu;
}

bool
DocanConfig::getEcuInfo(const uint16_t ecu, N_EcuInfo_t& info)
{
    bool ret = false;
    for (auto &it : can_ecu_info_list_) {
        if (it.address_logical == ecu) {
            info = it;
            ret = true;
            break;
        }
    }
    return ret;
}

const std::vector<N_EcuInfo_t>&
DocanConfig::getEcuInfoList()
{
    DOCAN_LOG_D("getEcuInfoList can_ecu_info_list_ size: %ld.", can_ecu_info_list_.size());
    return can_ecu_info_list_;
}

const std::vector<N_RouteInfo_t>&
DocanConfig::getRouteInfoList()
{
    DOCAN_LOG_D("getRouteInfoList can_route_info_list_ size: %ld.", can_route_info_list_.size());
    return can_route_info_list_;
}


} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
