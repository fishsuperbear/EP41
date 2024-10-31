/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "docan_event_receiver.h"
#include "diag/docan/manager/docan_task_runner.h"
#include "diag/docan/config/docan_config.h"
#include "diag/docan/manager/docan_sys_interface.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanEventReceiver* DocanEventReceiver::s_instance = nullptr;

    DocanEventReceiver::DocanEventReceiver()
    {
    }

    DocanEventReceiver::~DocanEventReceiver()
    {
    }

    DocanEventReceiver* DocanEventReceiver::instance()
    {
        if (nullptr == s_instance) {
            s_instance = new DocanEventReceiver();
        }
        return s_instance;
    }

    void DocanEventReceiver::destroy()
    {
        if (NULL != s_instance) {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    int32_t DocanEventReceiver::Init(void)
    {
        return 0;
    }

    int32_t DocanEventReceiver::Start(void)
    {
        route_info_list_ = DocanConfig::instance()->getRouteInfoList();
        return 0;
    }

    int32_t DocanEventReceiver::Stop(void)
    {
        return 0;
    }

    int32_t DocanEventReceiver::Deinit(void)
    {
        return 0;
    }

    int32_t DocanEventReceiver::ReceiveCanPacket(const N_EcuInfo_t& ecuInfo, const CanPacket& packet)
    {
        if (ecuInfo.diag_type == 2) {
            // 2: remote, can packet from can gw socket
            for (auto it : route_info_list_) {
                if (it.if_name == ecuInfo.if_name) {
                    if (DIAG_FUNCTIONAL_ADDR_DOCAN == packet.frame.can_id) {
                        // functional addr need forwad all the ecu
                        DocanSysInterface::instance()->AddAllCanPacket(packet);
                        break;
                    }

                    for (auto iter : it.forward_table) {
                        if (iter.gw_canid_tx == packet.frame.can_id) {
                            CanPacket fwpacket = packet;
                            fwpacket.frame.can_id = iter.forword_canid_tx;
                            DocanSysInterface::instance()->AddCanPacket(iter.forword_logical_addr, fwpacket);
                            break;
                        }
                    }
                    break;
                }
            }
        }
        else {
            // 1: local, default can packet from ecus
            if (DocanTaskRunner::instance()->isOperationRunning(DOCAN_NTTASK_SEND_COMMAND + ecuInfo.address_logical)) {
                // local request need response first
                uint32_t eventType = DOCAN_EVENT_ECU;
                uint32_t eventId = ecuInfo.address_logical;
                int32_t evtVal1 = packet.frame.can_id;
                int32_t evtVal2 = (uint8_t)((packet.frame.data[0] >> 4) & 0x0F);
                std::vector<uint8_t> evtData = std::vector<uint8_t>(packet.frame.data, packet.frame.data + 8);
                DocanTaskEvent *event = new DocanTaskEvent(eventType, eventId, evtVal1, evtVal2, evtData);
                onEvent(event);
            }
            else {
                // current uds request is not existed
                for (auto it : route_info_list_) {
                    for (auto iter : it.forward_table) {
                        if (iter.forword_canid_rx == packet.frame.can_id) {
                            // forward ecu response to gw
                            CanPacket fwpacket = packet;
                            fwpacket.frame.can_id = iter.forword_canid_rx;
                            DocanSysInterface::instance()->AddCanPacket(it.address_logical, fwpacket);
                            break;
                        }
                    }
                }
                // no forward canid match, regard as ecu response
                uint32_t eventType = DOCAN_EVENT_ECU;
                uint32_t eventId = ecuInfo.address_logical;
                int32_t evtVal1 = packet.frame.can_id;
                int32_t evtVal2 = (uint8_t)((packet.frame.data[0] >> 4) & 0x0F);
                std::vector<uint8_t> evtData = std::vector<uint8_t>(packet.frame.data, packet.frame.data + 8);
                DocanTaskEvent *event = new DocanTaskEvent(eventType, eventId, evtVal1, evtVal2, evtData);
                onEvent(event);
            }
        }
        return 0;
    }

    void DocanEventReceiver::onEvent(DocanTaskEvent* ev)
    {
        if (nullptr == ev) {
            return;
        }
        postEvent(ev);
    }

    void DocanEventReceiver::postEvent(DocanTaskEvent *ev)
    {
        if (nullptr == ev) {
            DOCAN_LOG_E("ev is nullptr!");
            return;
        }
        DocanTaskRunner::instance()->post(ev);
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */
