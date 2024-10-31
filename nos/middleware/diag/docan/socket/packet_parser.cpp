/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class PacketParser complement
 */

#include "packet_parser.h"
#include "diag/docan/taskbase/docan_task_event.h"
#include "diag/docan/manager/docan_event_receiver.h"
#include "diag/docan/common/docan_internal_def.h"
#include "diag/docan/log/docan_log.h"

namespace hozon {
namespace netaos {
namespace diag {


    PacketParser::PacketParser()
    {
    }

    PacketParser::~PacketParser()
    {
    }

    int32_t PacketParser::Init()
    {
        return 0;
    }

    int32_t PacketParser::Start()
    {
        return 0;
    }

    int32_t PacketParser::Stop()
    {
        return 0;
    }

    int32_t PacketParser::ParserCanPacket(const N_EcuInfo_t& ecuInfo, const CanPacket& packet)
    {
        // DOCAN_LOG_D("CanRaw ecu: %s, canif: %s, canid: %x, data: [%02X %02X %02X %02X %02X %02X %02X %02X]", socket.frame_id.c_str(), socket.if_name.c_str(), packet.frame.can_id, packet.frame.data[0], packet.frame.data[1]
        //     , packet.frame.data[2], packet.frame.data[3], packet.frame.data[4], packet.frame.data[5], packet.frame.data[6], packet.frame.data[7]);

        return DocanEventReceiver::instance()->ReceiveCanPacket(ecuInfo, packet);
    }

    int32_t PacketParser::ParserCanTPPacket(const CanTPSocketInfo& socket, const CanTPPacket& packet)
    {
        return 0;
    }


} // end of diag
} // end of netaos
} // end of hozon

/* EOF */
