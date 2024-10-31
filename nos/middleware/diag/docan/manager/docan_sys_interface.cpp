/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase implement
 */

#include "docan_sys_interface.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/docan/config/docan_config.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {

    DocanSysInterface* DocanSysInterface::s_instance = nullptr;

    DocanSysInterface* DocanSysInterface::instance()
    {
        if (nullptr == s_instance) {
            s_instance = new DocanSysInterface;
        }
        return s_instance;
    }

    void DocanSysInterface::destroy()
    {
        if (nullptr != s_instance) {
            delete s_instance;
            s_instance = nullptr;
        }
    }

    DocanSysInterface::DocanSysInterface()
    {
    }

    DocanSysInterface::~DocanSysInterface()
    {
    }

    int32_t DocanSysInterface::Init(void)
    {
        std::vector<N_EcuInfo_t> infoList = DocanConfig::instance()->getEcuInfoList();
        for (auto it: infoList) {
            std::shared_ptr<CanSocket> sock = std::make_shared<CanSocket>(it);
            sock->Init();
            socket_list_.push_back(sock);
        }

        std::vector<N_RouteInfo_t> routeList = DocanConfig::instance()->getRouteInfoList();
        for (auto iter : routeList) {
            // N_EcuInfo_t info;
            // info.ecu_name = iter.route_name;
            // info.if_name = iter.if_name;
            // info.address_logical = iter.address_logical;
            // info.diag_type = 2;  // GW
            // info.can_type = 2;  // GW
            // info.canid_rx = 0;
            // info.canid_tx = 0;
            // can_filter filter;
            // filter.can_id = 0x700;
            // filter.can_mask = 0x700;
            // info.filters.push_back(filter);
            // std::shared_ptr<CanSocket> sock = std::make_shared<CanSocket>(info);
            // socket_list_.push_back(sock);

            std::shared_ptr<CanGateway> routes = std::make_shared<CanGateway>(iter);
            routes->Init();
            route_list_.push_back(routes);
        }
        return 0;
    }

    int32_t DocanSysInterface::Start(void)
    {
        return 0;
    }

    int32_t DocanSysInterface::Stop(void)
    {
        for (auto it : socket_list_) {
            if (it->IsStop()) {
                it->Stop();
            }
        }
        return 0;
    }

    int32_t DocanSysInterface::Deinit(void)
    {
        return 0;
    }

    int32_t DocanSysInterface::AddCanPacket(uint16_t ecu, const CanPacket& packet)
    {
        DOCAN_LOG_D("AddCanPacket ecu: %X, canid: %X.", ecu, packet.frame.can_id);
        int32_t ret = -1;
        for (auto it : socket_list_) {
            if (it->GetEcuInfo().address_logical == ecu) {
                if (it->IsStop()) {
                    it->Start();
                }
                ret = it->AddSendPacket(packet);
                break;
            }
        }
        return ret;
    }

    int32_t DocanSysInterface::AddCanSendQueue(uint16_t ecu, const std::vector<CanPacket>& queue)
    {
        DOCAN_LOG_D("AddCanSendQueue ecu: %X, canid: %X, queue size: %ld.", ecu, queue[0].frame.can_id, queue.size());
        int32_t ret = -1;
        for (auto it : socket_list_) {
            if (it->GetEcuInfo().address_logical == ecu) {
                if (it->IsStop()) {
                    it->Start();
                }
                ret = it->AddSendQueue(queue);
                break;
            }
        }
        return ret;
    }

    int32_t DocanSysInterface::AddAllCanPacket(const CanPacket& packet)
    {
        int32_t ret = -1;
        for (auto it : socket_list_) {
            if (it->IsStop()) {
                it->Start();
            }
            ret = it->AddSendPacket(packet);
        }
        return ret;
    }

    int32_t DocanSysInterface::AddAllCanSendQueue(const std::vector<CanPacket>& queue)
    {
        DOCAN_LOG_D("AddAllCanSendQueue queue size: %ld.", queue.size());
        int32_t ret = -1;
        for (auto it : socket_list_) {
            if (it->IsStop()) {
                it->Start();
            }
            ret = it->AddSendQueue(queue);
        }
        return ret;
    }

    int32_t DocanSysInterface::StopChannel(uint16_t ecu)
    {
        int32_t ret = -1;
        for (auto it : socket_list_) {
            if (!it->IsStop()) {
                it->Stop();
            }
        }
        return ret;
    }

    bool DocanSysInterface::IsChannelStopped(uint16_t ecu)
    {
        bool ret = false;
        for (auto it : socket_list_) {
            if (it->GetEcuInfo().address_logical == ecu && it->IsStop()) {
                ret =  true;
            }
        }
        return ret;
    }

} // end of diag
} // end of netaos
} // end of hozon
/* EOF */