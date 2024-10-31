/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 *      Chunk info handler.
 * Create: 2022-01-23
 * Notes: NA
 */

#ifndef CODE_RTFCHUNKMANAGER_H
#define CODE_RTFCHUNKMANAGER_H

#include <memory>
#include <queue>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"

namespace rtf {

struct OutMessage {
    OutMessage() : msgBuff(nullptr), buffLen(0), keyValueLen(0)
    {}
    ara::core::Vector<ara::core::String> fieldName;
    ara::core::Vector<ara::core::String> fieldValue;
    uint8_t*                             msgBuff;
    uint32_t                             buffLen;
    uint32_t                             keyValueLen;
};

struct OutConnx {
    OutConnx() : keyValueLen(0)
    {}
    ara::core::Vector<ara::core::String> fieldName;
    ara::core::Vector<ara::core::String> fieldValue;
    uint32_t                             keyValueLen;
};

struct OutChunkHeader {
    OutChunkHeader() : keyValueLen(0)
    {}
    ara::core::Vector<ara::core::String> fieldName;
    ara::core::Vector<ara::core::String> fieldValue;
    uint32_t                             keyValueLen;
};

class RtfChunkManager {
public:
    RtfChunkManager();
    ~RtfChunkManager();
    bool SetChunkHeaderRecord(ara::core::Vector<ara::core::String> const& fieldName,
                              ara::core::Vector<ara::core::String> const& fieldValue);
    uint32_t GetChunkHeaderRecordLength() const;
    bool SetMessageRecord(ara::core::Vector<ara::core::String> const& fieldName,
                          ara::core::Vector<ara::core::String> const& fieldValue,
                          uint8_t* buf,
                          uint32_t length);

    bool SetChunkIndex(ara::core::Vector<ara::core::String> const& fieldName,
                       ara::core::Vector<ara::core::String> const& fieldValue,
                       uint8_t* buf,
                       uint32_t length);
    uint32_t GetChunkIndexSize() const;

    bool SetTempChunkInfo(ara::core::Vector<ara::core::String> const& fieldName,
                          ara::core::Vector<ara::core::String> const& fieldValue,
                          uint8_t* buf,
                          uint32_t length);
    uint32_t GetTempChunkInfoSize() const;

    uint32_t GetChunkSize() const;

    bool SetTempConnectionRecord(ara::core::Vector<ara::core::String> const& fieldName,
                                 ara::core::Vector<ara::core::String> const& fieldValue);

    uint32_t GetTempConnectionRecordLength() const;

    void GetChunk(OutChunkHeader& chunkHeader,
                  std::queue<OutMessage>& queueBuff,
                  std::queue<OutMessage>& queueChunkIndex,
                  OutMessage& tempChunkInfo,
                  std::queue<OutConnx>& queueConnx);
    std::queue<OutMessage> GetChunkMessages();

    bool SetcurrentChunkNewEventCount(const uint32_t& currentChunkNewEventCount);

    uint32_t GetcurrentChunkNewEventCount() const;
    uint32_t GetRecordHeaderLen(ara::core::Vector<ara::core::String> const& fieldName,
                                ara::core::Vector<ara::core::String> const& fieldValue) const;

    void DeleteResource() noexcept;
private:
    OutChunkHeader chunkHeader_;
    std::queue<OutMessage> queueBuff_;
    std::queue<OutMessage> queueChunkIndex_;
    OutMessage tempChunkInfo_;
    std::queue<OutConnx> queueConnx_;

    uint32_t chunkSize_;
    uint32_t chunkIndexSize_;
    uint32_t tempChunkInfoSize_;
    uint32_t connxSize_;
    uint32_t currentChunkNewEventCount_;
};
} // end of namespace rtf
#endif
