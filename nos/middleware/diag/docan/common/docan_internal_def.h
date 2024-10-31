/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  DoCan defination Header
 */

#ifndef DOCAN_INTERNAL_DEF_H_
#define DOCAN_INTERNAL_DEF_H_

#include <vector>
#include <string>
#include <linux/can.h>

namespace hozon {
namespace netaos {
namespace diag {

    /// define general data
const     uint16_t DIAG_CAN_FRAME_LENGTH                                                = 4096;   // Canfd frame size
const     uint16_t DIAG_ETHERNET_FRAME_LENGTH                                           = 1500;  // MTU ethernet
const     uint16_t DIAG_QUQUE_BUFF_SIZE                                                 = 1024;

/// ECU canid and ip  mac info
const     uint16_t    DIAG_FUNCTIONAL_ADDR_DOCAN    = 0x7DF;
const     uint16_t    DIAG_FUNCTIONAL_ADDR_DOIP     = 0xE400;

const     uint16_t    DIAG_LRR_CANID_REQUEST        = 0x7C5;
const     uint16_t    DIAG_LRR_CANID_RESPONSE       = 0x7D5;
const     uint16_t    DIAG_LRR_CONTI_CANID_REQUEST  = 0x791;
const     uint16_t    DIAG_LRR_CONTI_CANID_RESPONSE = 0x799;


typedef struct CanTPSocketInfo {
    std::string frame_id;
    std::string if_name;
    uint16_t canid_tx;
    uint16_t canid_rx;
} CanTPSocketInfo;

typedef struct CanTPPacket {
    uint32_t sec;
    uint32_t nsec;
    uint32_t len;
    uint8_t data[DIAG_CAN_FRAME_LENGTH] = { 0 };
} CanTPPacket;

typedef struct CanPacket {
    uint32_t sec;
    uint32_t nsec;
    uint32_t len;
    uint8_t STmin;
    can_frame frame;
} CanPacket;

// typedef enum {
//     N_Invalid = 0,
//     N_CAN_All        = 0x07DF,    // logical addr 0x7DF,
//     N_GW             = 0x0E80,    // logical addr 0x0E80,
//     N_Tester         = 0x1062,    // logical addr 0x1062,
//     N_IMU            = 0x10AC,    // logical addr 0x10AC,
//     N_MDC            = 0x10C3,    // logical addr 0x10C3,
//     N_CornerRadar_FL = 0x10C4,    // logical addr 0x10C4,
//     N_FrontRadar     = 0x10C5,    // logical addr 0x10C5,
//     N_CornerRadar_FR = 0x10C7,    // logical addr 0x10C7,
//     N_CornerRadar_RL = 0x10C8,    // logical addr 0x10C8,
//     N_CornerRadar_RR = 0x10C9,    // logical addr 0x10C9,
//     N_USSC           = 0x10CC,    // logical addr 0x10CC,
// } N_Ecu_t;

/**
* @brief network protocol control information type
*/
typedef enum {
    N_PCItype_SF = 0,   //single frame
    N_PCItype_FF = 1,   //first frame
    N_PCItype_CF = 2,   //consecutive frame
    N_PCItype_FC = 3,   //flow control
} N_PCItype_t;

typedef struct {
    std::string if_name;
    std::string ecu_name;
    uint8_t can_type;   // 0: invalid, 1: can, 2: canfd
    uint8_t diag_type;  // 0: invalid, 1: local diag, 2: remote diag
    uint8_t ta_type;    // 0: invalid, 1: physical addr, 2: functional addr
    uint16_t canid_tx;
    uint16_t canid_rx;
    uint16_t address_logical;
    uint16_t address_functional;
    uint8_t  BS;
    uint8_t  STmin;
    uint16_t N_WFTmax;
    uint16_t N_As;
    uint16_t N_Ar;
    uint16_t N_Bs;
    uint16_t N_Br;
    uint16_t N_Cs;
    uint16_t N_Cr;
    std::vector<can_filter> filters;
} N_EcuInfo_t;

typedef struct {
    uint16_t gw_canid_tx;
    uint16_t gw_canid_rx;
    uint16_t forword_logical_addr;
    std::string forword_ecu;
    std::string forword_if_name;
    uint16_t forword_canid_tx;
    uint16_t forword_canid_rx;
} N_ForwordInfo_t;

typedef struct {
    uint16_t address_logical;
    uint16_t address_functional;
    std::string route_name;
    std::string if_name;
    std::vector<N_ForwordInfo_t> forward_table;
} N_RouteInfo_t;


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_INTERNAL_DEF_H_
/* EOF */
