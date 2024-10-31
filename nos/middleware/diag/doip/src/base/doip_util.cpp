/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip util class
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>

#include "diag/doip/include/base/doip_util.h"
#include "diag/doip/include/config/doip_config.h"


namespace hozon {
namespace netaos {
namespace diag {


DoipUtil::DoipUtil() {
}


uint16_t
DoipUtil::DoipBswap16(uint16_t x) {
    return ((x & 0x00ff) << 8) | \
           ((x & 0xff00) >> 8);
}

uint32_t
DoipUtil::DoipBswap32(uint32_t x) {
    return ((x & 0xff000000U) >> 24) | \
           ((x & 0x00ff0000U) >> 8)  | \
           ((x & 0x0000ff00U) << 8)  | \
           ((x & 0x000000ffU) << 24);
}

int32_t
DoipUtil::DoipGetRandomValue(int32_t max_value) {
    struct timespec tn;
    clock_gettime(CLOCK_REALTIME, &tn);
    //srand((uint32_t)time(NULL));
    srand(tn.tv_nsec);
    int32_t random_value = rand() % (max_value) + 1;
    return random_value;
}

uint8_t
DoipUtil::IsFunctianalAddress(uint16_t logical_address) {
    doip_entity_config_t entity_config = DoIPConfig::Instance()->GetEntityConfig();
    bool result = std::any_of(entity_config.fa_list.begin(), entity_config.fa_list.end(), [&](uint16_t fa){
        return logical_address == fa;
    });

    return result;
}

uint8_t
DoipUtil::IsTestEquipmentAddress(uint16_t logical_address) {
    doip_entity_config_t entity_config = DoIPConfig::Instance()->GetEntityConfig();
    bool result = std::any_of(entity_config.sa_whitelist.begin(), entity_config.sa_whitelist.end(), [&](uint16_t sa){
        return logical_address == sa;
    });

    return result;
}

uint8_t
DoipUtil::IsInternalEquipmentAddress(uint16_t logical_address) {
    // TODO(ace): reserved
    return DOIP_TRUE;
}

doip_node_udp_table_t *
DoipUtil::DoipNodeUdpListFindByIpPort(const std::list<doip_node_udp_table_t*>& node_udp_tables, char* ip, uint16_t port) {
    for (auto node_udp_table : node_udp_tables) {
        if (0 == strcmp(node_udp_table->ip, ip) && node_udp_table->port == port) {
            return node_udp_table;
        }
    }

    return nullptr;
}

doip_node_tcp_table_t *
DoipUtil::DoipNodeTcpListFindByFd(const std::list<doip_node_tcp_table_t*>& node_tcp_tables, int32_t fd) {
    for (auto node_tcp_table : node_tcp_tables) {
        if (node_tcp_table->fd == fd) {
            return node_tcp_table;
        }
    }

    return nullptr;
}

doip_node_tcp_table_t *
DoipUtil::DoipNodeTcpListFindByLA(const std::list<doip_node_tcp_table_t*>& node_tcp_tables, uint16_t logical_address) {
    for (auto node_tcp_table : node_tcp_tables) {
        if (node_tcp_table->equip_logical_address == logical_address) {
            return node_tcp_table;
        }
    }

    return nullptr;
}

uint8_t
DoipUtil::DoipNodeTcpListRegistFdCount(const std::list<doip_node_tcp_table_t*>& node_tcp_tables) {
    return std::count_if(node_tcp_tables.begin(), node_tcp_tables.end(), [&](doip_node_tcp_table_t* node_tcp_table){
        return DOIP_CONNECT_STATE_REGISTERED_ROUTING_ACTIVE == node_tcp_table->connection_state;
    });
}

doip_equip_udp_table_t *
DoipUtil::DoipEquipUdpListFindByEID(const std::list<doip_equip_udp_table_t*>& equip_udp_tables, char* eid) {
    char eid_array[DOIP_EID_SIZE] = { 0 };
    memcpy(eid_array, eid, DOIP_EID_SIZE);
    for (auto equip_udp_table : equip_udp_tables) {
        if (0 == memcmp(equip_udp_table->eid, eid_array, DOIP_EID_SIZE)) {
            return equip_udp_table;
        }
    }

    return nullptr;
}

doip_equip_udp_table_t *
DoipUtil::DoipEquipUdpListFindByIpPort(const std::list<doip_equip_udp_table_t*>& equip_udp_tables, char* ip, uint16_t port) {
    for (auto equip_udp_table : equip_udp_tables) {
        if (0 == strcmp(equip_udp_table->ip, ip) && equip_udp_table->port == port) {
            return equip_udp_table;
        }
    }

    return nullptr;
}

doip_equip_tcp_table_t *
DoipUtil::DoipEquipTcpListFindByFd(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables, int32_t fd) {
    for (auto equip_tcp_table : equip_tcp_tables) {
        if (equip_tcp_table->fd == fd) {
            return equip_tcp_table;
        }
    }

    return nullptr;
}

doip_equip_tcp_table_t *
DoipUtil::DoipEquipTcpListFindByLA(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                  uint16_t equip_logical_address, uint16_t entity_logical_address) {
    for (auto equip_tcp_table : equip_tcp_tables) {
#ifdef BUILD_FOR_ORIN
        if (equip_tcp_table->equip_logical_address == equip_logical_address) {
            if ((entity_logical_address == 4293
             || entity_logical_address == 4292
             || entity_logical_address == 4295
             || entity_logical_address == 4296
             || entity_logical_address == 4297)
             && (0 == strcmp(equip_tcp_table->ip, "172.16.90.10"))) {
                return equip_tcp_table;
            } else if (equip_tcp_table->entity_logical_address == entity_logical_address) {
                return equip_tcp_table;
            }
        }
#else
        if (equip_tcp_table->entity_logical_address == entity_logical_address
            && equip_tcp_table->equip_logical_address == equip_logical_address) {
            return equip_tcp_table;
        }
#endif
    }

    return nullptr;
}

doip_equip_tcp_table_t *
DoipUtil::DoipEquipTcpListFindByIPandLA(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                        char* ip, uint16_t equip_logical_address) {
    for (auto equip_tcp_table : equip_tcp_tables) {
        if (0 == strcmp(equip_tcp_table->ip, ip) && equip_tcp_table->equip_logical_address == equip_logical_address) {
            return equip_tcp_table;
        }
    }

    return nullptr;
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
/* EOF */
