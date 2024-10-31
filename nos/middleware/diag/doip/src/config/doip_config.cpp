/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip config
 */
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <algorithm>
#include "json/json.h"

#include "diag/doip/include/config/doip_config.h"
#include "diag/doip/include/base/doip_util.h"
#include "diag/doip/include/base/doip_logger.h"


namespace hozon {
namespace netaos {
namespace diag {

DoIPConfig *DoIPConfig::instancePtr_ = nullptr;
std::mutex DoIPConfig::instance_mtx_;
const char DIDS_DATA_FILE_PATH[] = "/cfg/dids/dids.json";
const char DIDS_DATA_BACK_FILE_PATH[] = "/cfg/dids/dids.json_bak_1";

DoIPConfig *DoIPConfig::Instance() {
    if (nullptr == instancePtr_) {
        std::lock_guard<std::mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_) {
            instancePtr_ = new DoIPConfig();
        }
    }
    return instancePtr_;
}


DoIPConfig::DoIPConfig() {
    protocal_version_ = 0x02;
    char vin[DOIP_VIN_SIZE] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0X11};
    memcpy(vin_, vin, DOIP_VIN_SIZE);
}

DoIPConfig::~DoIPConfig() {
}

std::vector<std::string>
DoIPConfig::Split(const std::string& inputStr, const std::string& regexStr) {
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

std::string
DoIPConfig::GetVinNumber() {
    std::string filePath = "";
    if (0 == access(DIDS_DATA_FILE_PATH, F_OK)) {
        filePath = DIDS_DATA_FILE_PATH;
    } else {
        if (0 == access(DIDS_DATA_BACK_FILE_PATH, F_OK)) {
            filePath = DIDS_DATA_BACK_FILE_PATH;
        }
    }

    if ("" == filePath) {
        return "";
    }

    std::ifstream ifs;
    ifs.open(filePath, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        return "";
    }

    std::string vin = "";
    std::string str = "";
    bool bFind = false;
    while (getline(ifs, str)) {
        if (std::string::npos != str.find("F190")) {
            bFind = true;
            continue;
        }

        if (bFind && (std::string::npos != str.find("string"))) {
            auto vec = Split(str, "\"");
            if (vec.size() > 3) {
                vin = vec[3];
            }

            break;
        }
    }

    ifs.close();
    return vin;
}

bool
DoIPConfig::LoadConfig(std::string& doip_config) {
#ifdef BUILD_FOR_MDC
    std::string config_path = "/opt/usr/diag_update/mdc-llvm/conf/doip_config.json";
#elif BUILD_FOR_J5
    std::string config_path = "/userdata/diag_update/j5/conf/doip_config.json";
#elif BUILD_FOR_ORIN
    std::string config_path = "/app/runtime_service/diag_server/conf/doip_config.json";
#else
    std::string config_path = "/app/runtime_service/diag_server/conf/doip_config.json";
#endif

    if (doip_config == "") {
        doip_config = config_path;
    }

    std::string str = GetVinNumber();
    DOIP_INFO << "DoIPConfig::LoadConfig vin: " << str;
    memcpy(vin_, str.c_str(), std::min((int32_t)str.length(), DOIP_VIN_SIZE));

    DOIP_INFO << "DoIPConfig::LoadConfig configPath: " << doip_config.c_str();
    char* json = GetJsonAll(doip_config.c_str());
    if (NULL == json) {
        return false;
    }

    return ParseJSON(json);
}

char*
DoIPConfig::GetJsonAll(const char *fname) {
    FILE *fp;
    char *str;
    char txt[5000];
    int filesize;
    if ((fp = fopen(fname, "r")) == NULL) {
        DOIP_ERROR << "<DoIPConfig> open file " << fname << " fail!";
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    filesize = ftell(fp);

    str = reinterpret_cast<char*>(malloc(filesize + 1));  // malloc more size, or strcat will coredump
    if (!str) {
        DOIP_ERROR << "malloc error.";
        fclose(fp);
        return NULL;
    }
    memset(str, 0, filesize + 1);

    rewind(fp);
    while ((fgets(txt, 1000, fp)) != NULL) {
        strcat(str, txt);
    }
    fclose(fp);

    return str;
}

bool
DoIPConfig::ParseJSON(char* jsonstr) {
    Json::CharReaderBuilder readerBuilder;
    std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
    Json::Value  rootValue;
    JSONCPP_STRING errs;

    bool res = reader->parse(jsonstr, jsonstr + strlen(jsonstr), &rootValue, &errs);

    if (!res || !errs.empty()) {
        DOIP_ERROR << "<DoIPConfig> bad json format!";
        if (jsonstr != NULL) {
            free(jsonstr);
        }
        return false;
    }
    Json::Value & resultValue = rootValue["DoIpNetworkConfiguration"];
    for (uint32_t i = 0; i < resultValue.size(); i++) {
        doip_net_source_t* net_source = new doip_net_source_t;
        Json::Value subJson = resultValue[i];
        std::string netType = subJson["netType"].asString();
        if (netType == "IPv6") {
            net_source->net_type = DOIP_NET_TYPE_IPV6;
        } else {
            net_source->net_type = DOIP_NET_TYPE_IPV4;
        }

        net_source->source_type = subJson["sourceType"].asString();
        std::string if_use = subJson["if_use"].asString();
        if (if_use == "doipserver") {
            net_source->if_use = DOIP_IF_USE_DOIP_SERVER;
        } else if (if_use == "doipclient") {
            net_source->if_use = DOIP_IF_USE_DOIP_CLIENT;
        } else {
            net_source->if_use = DOIP_IF_USE_DOIP_BOTH;
        }

        net_source->if_name = subJson["ifName"].asString();
        net_source->ip = subJson["localIp"].asString();
        net_source->multicast_ip = subJson["multicastIp"].asString();
        net_source->tcp_port = subJson["tcpPort"].asUInt64();
        net_source->udp_port = subJson["udpPort"].asUInt64();

        DOIP_DEBUG << "<DoIPConfig> ParseJSON: netType is " << netType.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: sourceType is " << net_source->source_type.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: if_use is " << if_use.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: ifName is " << net_source->if_name.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: localIp is " << net_source->ip.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: multicastIp is " << net_source->multicast_ip.c_str();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: tcpPort is " << net_source->tcp_port;
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: udpPort is " << net_source->udp_port;

        net_source_list_.push_back(net_source);
    }

    resultValue = rootValue["DoIpRoutingTable"];
    for (uint32_t i = 0; i < resultValue.size(); i++) {
        Json::Value subJson = resultValue[i];
        routing_table_[subJson["logicalAddress"].asUInt()] = subJson["Ip"].asString();
        target_net_mds_[subJson["logicalAddress"].asUInt()] = subJson["maxRequestBytes"].asUInt64();
        ip_if_map_[subJson["Ip"].asString()] = subJson["ifName"].asString();
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: routing LA is " << subJson["logicalAddress"].asUInt() << ", IP is " <<  subJson["Ip"].asString().c_str() \
                   << ", IF is " << subJson["ifName"].asString().c_str() << ", mds is " << subJson["maxRequestBytes"].asString();
    }

    resultValue = rootValue["DoIpTimerConfiguration"];
    timer_config_.max_initial_vehicle_announcement_time = resultValue["maxInitialVehicleAnnouncementTime"].asUInt();
    timer_config_.interval_vehicle_announcement_time = resultValue["vehicleAnnouncementInterval"].asUInt();
    timer_config_.vehicle_announcement_count = resultValue["vehicleAnnouncementCount"].asUInt();
    timer_config_.tcp_initial_inactivity_time = resultValue["tcpInitialInactivityTime"].asUInt();
    timer_config_.tcp_general_inactivity_time = resultValue["tcpGeneralInactivityTime"].asUInt();
    timer_config_.tcp_alivecheck_time = resultValue["tcpAliveCheckResponseTimeout"].asUInt();
    timer_config_.doip_ack_time = resultValue["doipAckTimeout"].asUInt();

    DOIP_DEBUG << "<DoIPConfig> ParseJSON: max_initial_vehicle_announcement_time is " << timer_config_.max_initial_vehicle_announcement_time;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: interval_vehicle_announcement_time is " << timer_config_.interval_vehicle_announcement_time;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: vehicle_announcement_count is " << timer_config_.vehicle_announcement_count;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: tcp_initial_inactivity_time is " << timer_config_.tcp_initial_inactivity_time;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: tcp_general_inactivity_time is " << timer_config_.tcp_general_inactivity_time;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: tcp_alivecheck_time is " << timer_config_.tcp_alivecheck_time;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: doip_ack_time is " << timer_config_.doip_ack_time;


    resultValue = rootValue["DoIpEntityConfiguration"];
    std::string entity_type = resultValue["entity_type"].asString();
    if (entity_type == "edge_gateway") {
        entity_config_.entity_type = DOIP_ENTITY_TYPE_EDGE_GATEWAY;
    } else if (entity_type == "gateway") {
        entity_config_.entity_type = DOIP_ENTITY_TYPE_GATEWAY;
    } else {
        entity_config_.entity_type = DOIP_ENTITY_TYPE_NODE;
    }
    uint64_t eid_tmp = resultValue["eid"].asUInt64();
    uint64_t gid_tmp = resultValue["gid"].asUInt64();
    memcpy(eid_, &eid_tmp, sizeof(eid_));
    memcpy(gid_, &gid_tmp, sizeof(gid_));
    for (int i = 0; i < DOIP_EID_SIZE / 2; i++) {
        char tmp = eid_[DOIP_EID_SIZE - i - 1];
        eid_[DOIP_EID_SIZE - i - 1] = eid_[i];
        eid_[i] = tmp;
    }
    for (int i = 0; i < DOIP_GID_SIZE / 2; i++) {
        char tmp = gid_[DOIP_GID_SIZE - i - 1];
        gid_[DOIP_GID_SIZE - i - 1] = gid_[i];
        gid_[i] = tmp;
    }
    entity_config_.logical_address = resultValue["logicalAddress"].asUInt64();
    entity_config_.mcts = resultValue["maxTesterConnections"].asUInt();
    entity_config_.mds = resultValue["maxRequestBytes"].asUInt64();

    DOIP_DEBUG << "<DoIPConfig> ParseJSON: eid_ is " << eid_tmp;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: gid_ is " << gid_tmp;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: entity_type is " << entity_type.c_str();
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: logical_address is " << entity_config_.logical_address;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: mcts is " << entity_config_.mcts;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: mds is " << entity_config_.mds;

    Json::Value defaultValue;
    const auto &jsonValue = resultValue.get("sourceAddressWhiteList", defaultValue);
    for (Json::ArrayIndex i = 0; i != jsonValue.size(); ++i) {
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: sourceAddressWhiteList[" << i << "] : " << static_cast<uint16_t>(jsonValue[i].asUInt());
        entity_config_.sa_whitelist.emplace_back(static_cast<uint16_t>(jsonValue[i].asUInt()));
    }

    const auto &jsonValue2 = resultValue.get("functionAddressList", defaultValue);
    for (Json::ArrayIndex i = 0; i != jsonValue2.size(); ++i) {
        DOIP_DEBUG << "<DoIPConfig> ParseJSON: functionAddressList[" << i << "] : " << static_cast<uint16_t>(jsonValue2[i].asUInt());
        entity_config_.fa_list.emplace_back(static_cast<uint16_t>(jsonValue2[i].asUInt()));
    }


    resultValue = rootValue["DoIpSwitchConfiguration"];
    switch_config_.activation_line_dependence = resultValue["isActivationLineDependent"].asBool();
    switch_config_.resource_init_by_if = resultValue["resourceInitByIf"].asBool();
    switch_config_.vin_gid_sync_use = resultValue["vinGidSyncUse"].asBool();
    switch_config_.use_mac_as_eid = resultValue["eidUseMac"].asBool();
    switch_config_.further_acition_required = resultValue["furtherAcitionRequired"].asBool();
    switch_config_.entity_status_mds_use = resultValue["entityStatusMaxByteFieldUse"].asBool();
    switch_config_.power_mode_support = resultValue["power_mode_support"].asBool();
    switch_config_.authentication_required = resultValue["authentication_required"].asBool();
    switch_config_.confirmation_required = resultValue["confirmation_required"].asBool();

    DOIP_DEBUG << "<DoIPConfig> ParseJSON: activation_line_dependence is " << switch_config_.activation_line_dependence;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: resource_init_by_if is " << switch_config_.resource_init_by_if;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: vin_gid_sync_use is " << switch_config_.vin_gid_sync_use;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: use_mac_as_eid is " << switch_config_.use_mac_as_eid;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: further_acition_required is " << switch_config_.further_acition_required;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: entity_status_mds_use is " << switch_config_.entity_status_mds_use;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: power_mode_support is " << switch_config_.power_mode_support;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: authentication_required is " << switch_config_.authentication_required;
    DOIP_DEBUG << "<DoIPConfig> ParseJSON: confirmation_required is " << switch_config_.confirmation_required;

    if (jsonstr != NULL) {
        free(jsonstr);
    }

    return true;
}


void
DoIPConfig::SetVIN(char* vin, uint8_t len) {
    if (len != DOIP_VIN_SIZE || vin == nullptr) {
        return;
    }
    memcpy(vin_, vin, DOIP_VIN_SIZE);
}

char*
DoIPConfig::GetVIN() {
    return vin_;
}

void
DoIPConfig::SetGID(char* gid, uint8_t len) {
    if (len != DOIP_GID_SIZE || gid == nullptr) {
        return;
    }
    memcpy(gid_, gid, DOIP_GID_SIZE);
}

char*
DoIPConfig::GetGID() {
    return gid_;
}

void
DoIPConfig::SetEID(char* eid, uint8_t len) {
    if (len != DOIP_EID_SIZE || eid == nullptr) {
        return;
    }
    memcpy(eid_, eid, DOIP_EID_SIZE);
}

char*
DoIPConfig::GetEID() {
    return eid_;
}

char*
DoIPConfig::GetRoutingIp(uint16_t logical_address) {
    if (routing_table_.count(logical_address) > 0) {
        return const_cast<char*>(routing_table_[logical_address].c_str());
    }
    return nullptr;
}

std::string
DoIPConfig::GetIfNameByType(doip_if_use_t type) {
    for (auto& net_source : net_source_list_) {
        if (net_source->if_use == type) {
            return net_source->if_name;
        }
    }
    return "";
}

std::string
DoIPConfig::GetIfNameByIp(char* ip) {
    std::string ips = ip;
    if (ip_if_map_.count(ips) > 0) {
        return ip_if_map_[ips];
    }

    return "";
}

std::vector<doip_net_source_t*>&
DoIPConfig::GetNetSourceList() {
    return net_source_list_;
}

std::unordered_map<uint16_t, std::string>&
DoIPConfig::GetRoutingTable() {
    return routing_table_;
}

doip_timer_config_t&
DoIPConfig::GetTimerConfig() {
    return timer_config_;
}

doip_entity_config_t&
DoIPConfig::GetEntityConfig() {
    return entity_config_;
}

doip_switch_config_t&
DoIPConfig::GetSwitchConfig() {
    return switch_config_;
}

uint8_t
DoIPConfig::GetProtocalVersion() {
    return protocal_version_;
}

uint32_t
DoIPConfig::GetMaxRequestBytes(uint16_t logical_address) {
    if (target_net_mds_.count(logical_address) > 0) {
        return target_net_mds_[logical_address];
    }
    return 0xFFFFFFFF;
}



}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
