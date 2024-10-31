/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip util class
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_UTIL_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_UTIL_H_

#include <stdint.h>
#include <time.h>
#include <list>

#include "diag/doip/include/data_type/doip_def_internal.h"

namespace hozon {
namespace netaos {
namespace diag {


class DoipUtil {
 public:
    static DoipUtil &Instance() {
        static DoipUtil instance;
        return instance;
    }

    uint16_t DoipBswap16(uint16_t x);
    uint32_t DoipBswap32(uint32_t x);
    int32_t DoipGetRandomValue(int32_t max_value);
    uint8_t IsFunctianalAddress(uint16_t logical_address);
    uint8_t IsTestEquipmentAddress(uint16_t logical_address);
    uint8_t IsInternalEquipmentAddress(uint16_t logical_address);
    doip_node_udp_table_t* DoipNodeUdpListFindByIpPort(const std::list<doip_node_udp_table_t*>& node_udp_tables,
                                                       char* ip, uint16_t port);
    doip_node_tcp_table_t* DoipNodeTcpListFindByFd(const std::list<doip_node_tcp_table_t*>& node_tcp_tables, int32_t fd);
    doip_node_tcp_table_t* DoipNodeTcpListFindByLA(const std::list<doip_node_tcp_table_t*>& node_tcp_tables, uint16_t logical_address);
    uint8_t DoipNodeTcpListRegistFdCount(const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    doip_equip_udp_table_t* DoipEquipUdpListFindByEID(const std::list<doip_equip_udp_table_t*>& equip_udp_tables, char* eid);
    doip_equip_udp_table_t* DoipEquipUdpListFindByIpPort(const std::list<doip_equip_udp_table_t*>& equip_udp_tables, char* ip, uint16_t port);
    doip_equip_tcp_table_t* DoipEquipTcpListFindByFd(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables, int32_t fd);
    doip_equip_tcp_table_t* DoipEquipTcpListFindByLA(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                     uint16_t equip_logical_address, uint16_t entity_logical_address);
    doip_equip_tcp_table_t* DoipEquipTcpListFindByIPandLA(const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                          char* ip, uint16_t equip_logical_address);

 private:
    DoipUtil();
    DoipUtil(const DoipUtil &);
    DoipUtil & operator = (const DoipUtil &);
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_BASE_DOIP_UTIL_H_
/* EOF */
