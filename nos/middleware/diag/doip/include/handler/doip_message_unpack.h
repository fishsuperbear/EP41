/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip message unpacker
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_UNPACK_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_UNPACK_H_

#include <string>
#include <memory>
#include <functional>
#include <list>

#include "diag/common/include/timer_manager.h"
#include "diag/doip/include/data_type/doip_def_internal.h"
#include "diag/doip/include/config/doip_config.h"


namespace hozon {
namespace netaos {
namespace diag {

const uint8_t TYPE_RA_RES_STEP = 1;
const uint8_t TYPE_DIAG_ACK_STEP = 2;

class DoIPMessageHandler;

class DoIPMessageUnPack {
 public:
    static void RegistMessageTaskCallback(std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> callback,
                                          DoIPMessageHandler* handler, const std::shared_ptr<TimerManager>& timeManager);
    static void VehicleIndentifyWaitTimerCallback(void* data);
    static void AliveCheckTimerCallback(void* data);
    /************************************DoipUnpack******************************************/
    /************************************UDP Server******************************************/
    static void DoipUnpackHeaderNegativeAckFromUdpserver(doip_link_data_t *link_data);
    static void DoipUnpackVehicleIdentificationReqFromUdpserver(doip_link_data_t *link_data, const std::list<doip_node_udp_table_t*>& node_udp_tables);
    static void DoipUnpackVehicleIdentificationReqEidFromUdpserver(doip_link_data_t *link_data);
    static void DoipUnpackVehicleIdentificationReqVinFromUdpserver(doip_link_data_t *link_data);
    static void DoipUnpackEntityStatusReqFromUdpserver(doip_link_data_t *link_data);
    static void DoipUnpackPowerModeReqFromUdpserver(doip_link_data_t *link_data);
    static void DoipUnpackAnnouceOrIdentifyResFromUdpserver(doip_link_data_t *link_data);
    /****************************************************************************************/
    /************************************UDP Client******************************************/
    static void DoipUnpackHeaderNegativeAckFromUdpclient(doip_link_data_t *link_data);
    static void DoipUnpackAnnouceOrIdentifyResFromUdpclient(doip_link_data_t *link_data);
    static void DoipUnpackEntityStatusResFromUdpclient(doip_link_data_t *link_data);
    static void DoipUnpackPowerModeResFromUdpclient(doip_link_data_t *link_data);
    /****************************************************************************************/
    /************************************TCP Server******************************************/
    static void DoipUnpackHeaderNegativeAckFromTcpserver(doip_link_data_t *link_data);
    static void DoipUnpackRoutingActivationReqFromTcpserver(doip_link_data_t *link_data, const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackAliveCheckResFromTcpserver(doip_link_data_t *link_data, const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackDiagnosticFromTcpserver(doip_link_data_t* link_data,
                                                  const std::list<doip_node_tcp_table_t*>& node_tcp_tables,
                                                  const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables);
    static void PerformFunctionalDiagnostics(doip_link_data_t* link_data, uint16_t logical_source_address, uint16_t logical_target_address);
    static void DoipUnpackDiagPositiveAckFromTcpserver(doip_link_data_t *link_data, const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackDiagNegativeAckFromTcpserver(doip_link_data_t *link_data, const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    /****************************************************************************************/
    /************************************TCP Client******************************************/
    static void DoipUnpackHeaderNegativeAckFromTcpclient(doip_link_data_t *link_data);
    static void DoipUnpackRoutingActivationResFromTcpclient(doip_link_data_t *link_data,
                                                            const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                            const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackAliveCheckReqFromTcpclient(doip_link_data_t *link_data, const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables);
    static void DoipUnpackDiagnosticFromTcpclient(doip_link_data_t *link_data,
                                                  const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                  const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackDiagPositiveAckFromTcpclient(doip_link_data_t *link_data,
                                                       const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                       const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    static void DoipUnpackDiagNegativeAckFromTcpclient(doip_link_data_t *link_data,
                                                       const std::list<doip_equip_tcp_table_t*>& equip_tcp_tables,
                                                       const std::list<doip_node_tcp_table_t*>& node_tcp_tables);
    /****************************************************************************************/
    /****************************************************************************************/

    static int32_t StartDoipReqTimer(doip_equip_tcp_table_t *equip_tcp_table, uint8_t type);
    static int32_t ReStartDoipReqTimer(doip_equip_tcp_table_t *equip_tcp_table);
    static int32_t StopDoipReqTimer(doip_equip_tcp_table_t *equip_tcp_table, uint8_t type);
    static void DoIPReqTimerCallback(void* data);

 private:
    static std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> messageTaskCallback_;
    static DoIPMessageHandler* handler_;
    static doip_alive_check_originator_t doip_alive_check_originator_;
    static std::shared_ptr<TimerManager> timerManager_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_UNPACK_H_
