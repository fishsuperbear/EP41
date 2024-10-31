/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip config
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_CONFIG_DOIP_CONFIG_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_CONFIG_DOIP_CONFIG_H_

#include <stdint.h>
#include <mutex>
#include <vector>
#include <string>
#include <unordered_map>

#include "diag/doip/include/data_type/doip_def_internal.h"


namespace hozon {
namespace netaos {
namespace diag {

typedef enum DOIP_NET_TYPE {
    DOIP_NET_TYPE_IPV4 = 0x00,
    DOIP_NET_TYPE_IPV6 = 0x01
} doip_net_type_t;

typedef enum DOIP_IF_USE {
    DOIP_IF_USE_DOIP_CLIENT    = 0x00,
    DOIP_IF_USE_DOIP_SERVER    = 0x01,
    DOIP_IF_USE_DOIP_BOTH      = 0x02
} doip_if_use_t;

typedef struct doip_net_source {
    doip_net_type_t net_type;
    std::string source_type;
    std::string if_name;
    int32_t priority;
    doip_if_use_t if_use;
    uint8_t announce_count;
    int32_t announce_wait_timer_fd;
    int32_t announce_interval_timer_fd;
    std::string ip;
    std::string multicast_ip;
    uint16_t tcp_port;
    uint16_t udp_port;
    uint8_t link_status;
    uint8_t ip_status;

    int32_t fd;
} doip_net_source_t;

typedef struct doip_timer_config {
    int32_t max_initial_vehicle_announcement_time;
    int32_t interval_vehicle_announcement_time;
    int32_t vehicle_announcement_count;
    int32_t tcp_initial_inactivity_time;
    int32_t tcp_general_inactivity_time;
    int32_t tcp_alivecheck_time;
    int32_t doip_ack_time;
} doip_timer_config_t;

typedef struct doip_entity_config {
    doip_entity_type_t entity_type;
    uint16_t logical_address;
    std::vector<uint16_t> sa_whitelist;
    std::vector<uint16_t> fa_list;
    uint8_t mcts;
    uint32_t mds;
} doip_entity_config_t;

typedef struct doip_switch_config {
    bool activation_line_dependence;
    bool resource_init_by_if;
    bool vin_gid_sync_use;
    bool use_mac_as_eid;
    bool further_acition_required;
    bool entity_status_mds_use;
    bool power_mode_support;
    bool authentication_required;
    bool confirmation_required;
} doip_switch_config_t;


class DoIPConfig {
 public:
    static DoIPConfig *Instance();
    bool LoadConfig(std::string& doip_config);
    void SetVIN(char* vin, uint8_t len);
    char* GetVIN();
    void SetGID(char* gid, uint8_t len);
    char* GetGID();
    void SetEID(char* eid, uint8_t len);
    char* GetEID();
    char* GetRoutingIp(uint16_t logical_address);

    std::string GetIfNameByType(doip_if_use_t type);
    std::string GetIfNameByIp(char* ip);
    std::vector<doip_net_source_t*>& GetNetSourceList();
    std::unordered_map<uint16_t, std::string>& GetRoutingTable();
    doip_timer_config_t& GetTimerConfig();
    doip_entity_config_t& GetEntityConfig();
    doip_switch_config_t& GetSwitchConfig();
    uint8_t GetProtocalVersion();
    uint32_t GetMaxRequestBytes(uint16_t logical_address);

 private:
    DoIPConfig();
    ~DoIPConfig();
    DoIPConfig(const DoIPConfig &);
    DoIPConfig & operator = (const DoIPConfig &);
    char* GetJsonAll(const char *fname);
    bool ParseJSON(char* jsonstr);
    std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr);
    std::string GetVinNumber();

    static DoIPConfig *instancePtr_;
    static std::mutex instance_mtx_;

    std::vector<doip_net_source_t*> net_source_list_;
    doip_timer_config_t timer_config_;
    doip_entity_config_t entity_config_;
    doip_switch_config_t switch_config_;

    uint8_t protocal_version_;
    char vin_[DOIP_VIN_SIZE];
    char eid_[DOIP_EID_SIZE];
    char gid_[DOIP_GID_SIZE];
    std::unordered_map<uint16_t, std::string> routing_table_;
    std::unordered_map<uint16_t, uint32_t> target_net_mds_;
    std::unordered_map<std::string, std::string> ip_if_map_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_CONFIG_DOIP_CONFIG_H_
