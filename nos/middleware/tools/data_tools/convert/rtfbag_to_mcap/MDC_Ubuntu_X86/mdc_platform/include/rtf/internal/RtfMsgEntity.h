/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class RtfMsgEntity.
 *      MessageEntity is deisgned to store message
 * Create: 2019-11-30
 * Notes: NA
 */
#ifndef RTF_MSG_ENTITY_H
#define RTF_MSG_ENTITY_H

#include <functional>
#include <set>

#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/internal/RtfBagFile.h"
#include "rtf/internal/RtfBagStructs.h"
#include "rtf/internal/RtfBuffer.h"

namespace rtf {
namespace rtfbag {
struct RtfTrueQuery {
    bool operator()(Connection const&) const
    {
        return true;
    }
};

struct RtfEventQuery {
    explicit RtfEventQuery(ara::core::String const& event);
    explicit RtfEventQuery(ara::core::Vector<ara::core::String> const& events,
                           ara::core::Vector<ara::core::String> const& skipEvents = {});
    bool IsSkipEvent(Connection const& info) const;

    bool operator()(Connection const& info) const;
    ara::core::Vector<ara::core::String> events_;
    ara::core::Vector<ara::core::String> skipEvents_;
};

struct RtfQuery {
    RtfQuery(std::function<bool(Connection const&)> const& query, uint64_t const& startTime, uint64_t const& endTime);

    std::function<bool(Connection const&)> query_;
    uint64_t startTime_;
    uint64_t endTime_;
};

struct RtfBagQuery {
    RtfBagQuery(RtfBagFile& bag, RtfQuery const& query);

    RtfBagFile* bag_;
    RtfQuery query_;
};

struct MessageRange	{
    MessageRange(std::multiset<MessageIndex>::const_iterator const& begin,
        std::multiset<MessageIndex>::const_iterator const& end,
        Connection const& connection, RtfBagQuery& bagQuery);
    std::multiset<MessageIndex>::const_iterator begin_;
    std::multiset<MessageIndex>::const_iterator end_;
    Connection const *connection_;
    RtfBagQuery* bagQuery_;
};

struct ViewIterHelper {
    ViewIterHelper(std::multiset<MessageIndex>::const_iterator const& iter, MessageRange& range);
    std::multiset<MessageIndex>::const_iterator iter_;
    MessageRange *range_;
};

struct ViewIterHelpCompare {
    bool operator()(ViewIterHelper const& first, ViewIterHelper const& second) const;
};

class RtfMsgEntity {
public:
    RtfMsgEntity(Connection const& conn, MessageIndex const& msgIndex, RtfBagFile& bag);
    ~RtfMsgEntity();

    FileErrorCode WriteMsg(RtfBuffer& buffer, const std::uint32_t len = 0) const;
    ara::core::String GetEvent() const;
    ara::core::String GetDataType() const;
    uint64_t GetTime() const;
    bool GetMsgSize(uint32_t& size) const;
    const Connection& GetConnection() const;
    uint32_t GetBagVersion() const;

private:
    Connection const *conn_;
    MessageIndex const *msgIndex_;
    RtfBagFile* bag_;
};
}  // namespace rtfbag
}  // namespace rtf
#endif
