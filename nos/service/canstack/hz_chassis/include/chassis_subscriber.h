/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface monitor[chassis]
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <shared_mutex>

#include <cstring>
#include "canbus_writer.h"
#include "cm/include/proxy.h"
#include "ep40_canfd_mcu_soc_v4_1_.h"
#include "hz_canagent.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "subscriber.h"
#include "proto/fsm/function_manager.pb.h"
#include "proto/hmi/avp.pb.h"
#include "proto/planning/warning.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "proto/perception/perception_parking_lot.pb.h"
#include "can_parser_chassis.h"


namespace hozon {
namespace netaos {
namespace canstack {
namespace chassis {

extern std::condition_variable canfd_msg_135_cv_;

class ChassisSubscriberCanFdMsg {
   public:
    inline void BitAssign_Byte(uint8_t& dst, uint8_t val, uint8_t mask) {
        val = val & mask;
        dst &= ~mask;
        dst = dst | val;
    }

    // void Set(::hozon::canfdmsg::CanFdmsgFrame canfdmsgframe);

    /**  将单纯的处理过的canfdmsg赋值到对应要发送的can帧中
    @param[out] wriete_canfd_frame 准备发出的can帧
    @param[in]  process_canfd_id_msg 处理过的msg
    @param[in]  bit_size 拷贝的字节数
    @return     void 无
    @note       将处理过的msg赋值到对应要发送的can帧中
    */
    void DataAssign(canfd_frame& wriete_canfd_frame, uint8_t* process_canfd_id_msg, const uint8_t bit_size) {
        std::lock_guard<std::mutex> lock(canfdmsg_mutex_);
        std::memcpy(wriete_canfd_frame.data, process_canfd_id_msg, bit_size);  // DataAssign
    }

    /**  将纠缠的处理过的canfdmsg赋值到对应要发送的can帧中
    @param[out] wriete_canfd_frame 准备发出的can帧
    @param[in]  process_canfd_id_msg 处理过的msg
    @param[in]  bit_size 拷贝的字节数
    @param[in]  mask 算法的掩码
    @return     void 无
    @note       将处理过的msg赋值到对应要发送的can帧中
    */
    void DataAssign(canfd_frame& wriete_canfd_frame, uint8_t* process_canfd_id_msg, const uint8_t bit_size, uint8_t* mask) {
        std::lock_guard<std::mutex> lock(canfdmsg_mutex_);
        for (size_t i = 0; i < (sizeof(wriete_canfd_frame.data) / sizeof(wriete_canfd_frame.data[0])); i++) {
            process_canfd_id_msg[i] &= ~mask[i];
            // std::cout << "process_canfd_id_msg[" << i << "]: " << process_canfd_id_msg[i] << std::endl;
            wriete_canfd_frame.data[i] = process_canfd_id_msg[i] | (wriete_canfd_frame.data[i] & mask[i]);
        }
    }

    uint8_t can_120_frame_mask_[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                       0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x3F, 0x3F, 0x3F, 0x00, 0x00};

    uint8_t can_233_frame_mask_[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F, 0xFF, 0xFF, 0xFF, 0xFC, 0x03, 0xF0, 0x00,
                                       0xC0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00};

    uint8_t can_102_frame_mask_[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xF3, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00};

    uint8_t can_135_frame_mask_[16] = {0x00, 0x33, 0x20, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

    uint8_t process_canfd_102_msg[32] = {0};
    uint8_t process_canfd_120_msg[32] = {0};
    uint8_t process_canfd_1EC_msg[64] = {0};
    uint8_t process_canfd_1ED_msg[64] = {0};
    uint8_t process_canfd_211_msg[8] = {0};
    uint8_t process_canfd_230_msg[64] = {0};
    uint8_t process_canfd_231_msg[64] = {0};
    uint8_t process_canfd_232_msg[64] = {0};
    uint8_t process_canfd_233_msg[64] = {0};
    uint8_t process_canfd_234_msg[64] = {0};
    uint8_t process_canfd_264_msg[8] = {0};
    uint8_t process_canfd_135_msg[16] = {0};

   private:
    std::mutex canfdmsg_mutex_;
};

class ChassisSubscriber : public hozon::netaos::canstack::Subscriber {
   public:
    static ChassisSubscriber* Instance();
    virtual ~ChassisSubscriber();

    int Init() override;
    void Sub() override;
    int Stop() override;
    void Clear();

   private:
    ChassisSubscriber();

    void ChassisEgoHmiInfoCallback();
    void ChassisEgoHmiParkInfoReceive();
    void ChassisEgo2McuInfoReceive();
    void ChassisNNPEgo2McuInfoReceive();
    void ChassisStateMachineInfoReceive();
    void ChassisParkingLotInfoReceive();
    // void ChassisCanFdMsg1InfoReceive(std::shared_ptr<ProxyCanFdMsg> proxy_msg, uint8_t instanceid);
    void MergeCanFdMsg();
    void DataAssignEgoHmi(const std::shared_ptr<hozon::planning::WarningOutput>);
    void DataAssignEgoHmiPark(const std::shared_ptr<hozon::hmi::AvpToHmi>);
    void DataAssignEgo2Mcu(const std::shared_ptr<hozon::functionmanager::FunctionManagerOut>);
    void DataAssignStateMachine2Mcu(const std::shared_ptr<hozon::state::StateMachine>);
    void DataAssignParkingLot2Mcu(const std::shared_ptr<hozon::perception::ParkingLotOutArray>);

    void DataAssignCanFdMsg();

    int send_fd_;
    std::string canName_;
    bool serving_;
    static ChassisSubscriber* sinstance_;
    std::vector<canfd_frame> common_canfd_msg_vector_;
    bool _135_data_has_changed_flag_;  // true:数据发生了改变，false数据没有改变
    std::thread Can10msTask_thread_;

    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_102_t> _0x102_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_120_t> _0x120_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_180_t> _0x180_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_211_t> _0x211_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_230_t> _0x230_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_231_t> _0x231_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_232_t> _0x232_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_233_t> _0x233_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_234_t> _0x234_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_264_t> _0x264_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_event_135_t> _0x135_data_;
    std::shared_ptr<struct ep40_canfd_mcu_soc_v4_1_adcs_soc_mcu_rx_cyc_181_t> _0x181_data_;

    std::shared_timed_mutex data_mutex_ego2hmi_;
    std::shared_timed_mutex data_mutex_ego2hmi_park_;
    std::mutex data_mutex_ego2mcu_;
    std::mutex data_mutex_state_machine_;
    std::mutex data_mutex_parking_lot_;

    ChassisSubscriberCanFdMsg canfdmsg_sub_;

    canfd_frame ego_to_mcu_0x180_;     // 100ms
    canfd_frame state_machine_0x181_;  // 50ms

    /* 涉及到多个进程处理 */
    canfd_frame ego_hmi_frame_0x120_;  // 20ms
    canfd_frame ego_hmi_frame_0x233_;  // 50ms
    /* 涉及到多个进程处理 */

    canfd_frame canfd_msg_102_;  // 10ms
    canfd_frame canfd_msg_211_;  // 100ms
    canfd_frame canfd_msg_230_;  // 50ms
    canfd_frame canfd_msg_231_;  // 50ms
    canfd_frame canfd_msg_232_;  // 50ms
    canfd_frame canfd_msg_234_;  // 50ms
    canfd_frame canfd_msg_264_;  // 100ms
    canfd_frame canfd_msg_1EC_;  // 50ms
    canfd_frame canfd_msg_1ED_;  // 50ms

    canfd_frame ego_hmi_frame_0x135_;  // Event

    /* Tangled with canfdmsg */
    ChassisSubscriberCanFdMsg canfdmsg_sub_120_;
    ChassisSubscriberCanFdMsg canfdmsg_sub_233_;
    /* Tangled with canfdmsg */

    std::shared_ptr<CmProtoBufPubSubType> pub_sub_type_;
    std::shared_ptr<hozon::netaos::cm::Proxy> ego2hmi_proxy_;
    std::shared_ptr<hozon::netaos::cm::Proxy> ego2hmi_park_proxy_;
    std::shared_ptr<hozon::netaos::cm::Proxy> ego2mcu_proxy_;
    std::shared_ptr<hozon::netaos::cm::Proxy> nnp_ego2mcu_proxy_;
    std::shared_ptr<hozon::netaos::cm::Proxy> sm_to_mcu_proxy_;
    std::shared_ptr<hozon::netaos::cm::Proxy> perception_parking_lot_proxy_;
    // std::shared_ptr<hozon::netaos::cm::Proxy> canfd_proxy_;
};
}  // namespace chassis
}  // namespace canstack
}  // namespace netaos
}  // namespace hozon
