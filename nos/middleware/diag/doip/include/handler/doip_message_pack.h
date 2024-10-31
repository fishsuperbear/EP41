/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip message packer
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_PACK_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_PACK_H_

#include <string>
#include <memory>
#include <functional>

#include "diag/doip/include/data_type/doip_def_internal.h"
#include "diag/doip/include/config/doip_config.h"


namespace hozon {
namespace netaos {
namespace diag {

class DoIPMessageHandler;

class DoIPMessagePack {
 public:
    static void RegistMessageTaskCallback(std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> callback,
                                          DoIPMessageHandler* handler);

    /*************************************DoipPack*******************************************/
    static void DoipPackDiagnostic2Cache(doip_link_data_t *link_data, doip_equip_tcp_table_t* equip_tcp_table);
    static void DoipPackDiagnostic2Cache(doip_link_data_t *link_data, doip_equip_tcp_table_t* equip_tcp_table, uint16_t replace_address);
    /****************************************UDP*********************************************/
    static void DoipPackHeaderNegativeAck2Udp(doip_link_data_t *link_data, uint8_t code);
    static void DoipPackVehicleIdentifyReq2Udp(doip_link_data_t *link_data);
    static void DoipPackVehicleIdentifyReqWithEID2Udp(doip_link_data_t *link_data, char *eid);
    static void DoipPackVehicleIdentifyReqWithVIN2Udp(doip_link_data_t *link_data, char *vin);
    static void DoipPackAnnouceOrIdentifyResponse2Udp(doip_link_data_t *link_data,
                                                      uint8_t further_action,
                                                      uint8_t vin_gid_sync);
    static void DoipPackEntityStatusReq2Udp(doip_link_data_t *link_data);
    static void DoipPackEntityStatusRes2Udp(doip_link_data_t *link_data,
                                            const doip_entity_config_t& entity_config,
                                            uint8_t ncts);
    static void DoipPackPowerModeReq2Udp(doip_link_data_t *link_data);
    static void DoipPackPowerModeRes2Udp(doip_link_data_t *link_data, uint8_t diag_power_mode);

    /****************************************TCP*********************************************/
    static void DoipPackHeaderNegativeAck2Tcp(doip_link_data_t *link_data, uint8_t code);
    static void DoipPackRoutingActivationReq2Tcp(doip_link_data_t *link_data, uint16_t logical_source_address,
                                                 uint8_t activation_type, uint32_t oem_specific_use);
    static uint32_t DoipPackRoutingActivationRes2Tcp(doip_link_data_t *link_data,
                                                     uint16_t equip_logical_address,
                                                     uint16_t entity_logical_address,
                                                     uint8_t res_code,
                                                     uint32_t oem_specific_use);
    static void DoipPackAliveCheckReq2Tcp(doip_link_data_t *link_data);
    static void DoipPackAliveCheckRes2Tcp(doip_link_data_t *link_data, uint16_t equip_logical_address);
    static void DoipPackDiagnostic2Tcp(doip_link_data_t *link_data,
                                       uint16_t logical_source_address,
                                       uint16_t logical_target_address);
    static uint32_t DoipPackDiagPositiveAck2Tcp(doip_link_data_t *link_data,
                                            uint16_t source_logical_address,
                                            uint16_t target_logical_address);
    static uint32_t DoipPackDiagNegativeAck2Tcp(doip_link_data_t *link_data,
                                                uint16_t source_logical_address,
                                                uint16_t target_logical_address,
                                                uint8_t code);
    /****************************************************************************************/

 private:
    static std::function<uint32_t(doip_link_data_t*, std::string, uint32_t)> messageTaskCallback_;
    static DoIPMessageHandler* handler_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_HANDLER_DOIP_MESSAGE_PACK_H_
