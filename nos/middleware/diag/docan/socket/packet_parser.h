/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class PacketParser Header
 */

#ifndef DOCAN_PACKET_PARSER_H_
#define DOCAN_PACKET_PARSER_H_

#include "diag/docan/common/docan_internal_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class PacketParser {
public:
    PacketParser();
    virtual ~PacketParser();

    int32_t Init();
    int32_t Start();
    int32_t Stop();

    int32_t ParserCanPacket(const N_EcuInfo_t& ecuInfo, const CanPacket& packet);

    int32_t ParserCanTPPacket(const CanTPSocketInfo& socket, const CanTPPacket& packet);

private:
    PacketParser(const PacketParser&);
    PacketParser& operator=(const PacketParser&);

private:

    bool stop_;
};


} // end of diag
} // end of netaos
} // end of hozon

#endif   // DOCAN_PACKET_PARSER_H_
