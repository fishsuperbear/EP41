/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class RtfView.
 *      RtfView is for publish messages
 * Create: 2019-11-30
 * Notes: NA
 */
#ifndef RTF_VIEW_H
#define RTF_VIEW_H

#include <memory>

#include "ara/core/string.h"
#include "ara/core/vector.h"
#include "rtf/internal/RtfBagFile.h"
#include "rtf/internal/RtfBagStructs.h"
#include "rtf/internal/RtfMsgEntity.h"

namespace rtf {
namespace rtfbag {
constexpr uint64_t TIME_MIN = 0U;
constexpr uint64_t TIME_MAX = UINT64_MAX;

class RtfView {
public:
    class Iterator {
        friend class RtfView;
    public:
        explicit Iterator(RtfView const& view);
        ~Iterator();

        Iterator& operator++()
        {
            (void) Increase();
            return *this;
        }

        RtfMsgEntity& Value();
        uint64_t BeginTime();
        bool IsEnd() const;
        RtfMsgEntity GetPosValue(const uint32_t pos);
    protected:
        bool Populate();
        bool Increase();
        RtfMsgEntity& Dereference();
    private:
        RtfView const *view_;
        uint32_t size_;
        RtfMsgEntity* msgEntity_;
        ara::core::Vector<ViewIterHelper> iters_;
    };

    RtfView();
    ~RtfView();

    Iterator Begin() const;

    void AddQuery(RtfBagFile& bag, uint64_t const& startTime = TIME_MIN, uint64_t const& endTime = TIME_MAX);
    void AddQuery(RtfBagFile& bag, std::function<bool(Connection const&)> const& query,
        uint64_t const& startTime = TIME_MIN, uint64_t const& endTime = TIME_MAX);
    ara::core::Vector<Connection const*> GetConnections();
    uint32_t Size();
    uint64_t GetBeginTime();
    uint64_t GetEndTime();

protected:
    void UpdateQueries(RtfBagQuery& bagQuery);
    RtfMsgEntity* CreateMsgEntity(Connection const& connection,
        MessageIndex const& msgIndex, RtfBagFile& bag) const;

private:
    ara::core::Vector<std::shared_ptr<MessageRange>> ranges_;
    ara::core::Vector<std::shared_ptr<RtfBagQuery>>  queries_;
    uint64_t                         size_;
};
}  // namespace rtfbag
}  // namespace rtf
#endif
