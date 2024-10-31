/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: Structs definition of RtfBagFile class
 * Create: 2019-12-3
 */
#ifndef RTF_BAG_STRUCTS_H
#define RTF_BAG_STRUCTS_H

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/maintaind/impl_type_eventinfo.h"
#include "rtf/maintaind/impl_type_drivertype.h"
#include "rtf/maintaind/impl_type_serializeconfig.h"

namespace rtf {
namespace rtfbag {
struct BagFileHeader {
    BagFileHeader() : connectionCount(0), chunkCount(0), connectionPos(0)
    {}
    uint32_t connectionCount;
    uint32_t chunkCount;
    uint64_t connectionPos;
};

struct ChunkHeader {
    ChunkHeader(): chunkSize(0),
                   chunkCompressedSize(0),
                   connectionCount(0),
                   chunkInfoRecordPos(0),
                   nextChunkPos(0),
                   chunkStatus(0)
    {}
    uint32_t chunkSize;  // size of the chunk in bytes
    uint32_t chunkCompressedSize;  // size of the chunk in bytes
    uint32_t connectionCount;
    uint64_t chunkInfoRecordPos;
    uint64_t nextChunkPos;
    uint32_t chunkStatus;
};

struct ConnectionHeader {
    ConnectionHeader() : id(0)
    {}
    uint32_t id;
    ara::core::String event;
};

struct Connection {
    Connection() : type(""),
                   msgDef(""),
                   driverType(rtf::maintaind::DriverType::DDS),
                   startTime(UINT64_MAX),
                   endTime(0),
                   msgCount(0)
    {}
    ConnectionHeader header;
    ara::core::String type;
    ara::core::String msgDef;
    rtf::maintaind::DriverType driverType;
    rtf::maintaind::EventInfo eventInfo {};
    rtf::maintaind::SerializeConfig serializeConfig {};
    uint64_t startTime;
    uint64_t endTime;
    uint64_t msgCount;
};

struct ChunkConSumInfo {
    ChunkConSumInfo() : startTime(UINT64_MAX),
                        endTime(0),
                        msgCount(0)
    {}
    uint64_t startTime;
    uint64_t endTime;
    uint64_t msgCount;
};

struct MessageIndex {
    MessageIndex() : time(0), msgPos(0), chunkDataPos(0)
    {}
    uint64_t time;    // timestamp of the message
    uint64_t msgPos;  // absolute pos (byte) of the message record
    uint64_t chunkDataPos;
    bool operator<(MessageIndex const& msgIndex) const
    {
        return time < msgIndex.time;
    }
};

struct ChunkInfoHeader {
    ChunkInfoHeader() : connectionCount(0), startTime(0), endTime(0), chunkPos(0)
    {}
    uint32_t connectionCount;
    uint64_t startTime;
    uint64_t endTime;
    uint64_t chunkPos;
};

struct ChunkInfo {
    ChunkInfo() {}
    ChunkInfoHeader header;
    ara::core::Map<uint32_t, uint32_t> connectionIdCount;
};

struct EventMsg {
    EventMsg() : eventName(""), dataType(""), msgDef(""), driverType(rtf::maintaind::DriverType::DDS) {}
    ara::core::String eventName;
    ara::core::String dataType;
    ara::core::String msgDef;
    rtf::maintaind::DriverType driverType {};
    rtf::maintaind::EventInfo eventInfo {};
    rtf::maintaind::SerializeConfig serializeConfig {};
};

constexpr uint8_t MESSAGE_OP = 0x02U;
constexpr uint8_t FLIE_HEADER_OP = 0x03U;
constexpr uint8_t CHUNK_INDEX_OP = 0x04U;
constexpr uint8_t CHUNK_HEADER_OP = 0x05U;
constexpr uint8_t CHUNK_INFO_OP = 0x06U;
constexpr uint8_t CONNECTION_OP = 0x07U;
constexpr uint8_t CHUNK_END_OP = 0x08U;
constexpr uint32_t RATE_OF_BIT = 1024U;
constexpr uint32_t INIT_BAG_VERSION = 100U;
}  // namespace rtfbag
}  // namespace rtf
#endif