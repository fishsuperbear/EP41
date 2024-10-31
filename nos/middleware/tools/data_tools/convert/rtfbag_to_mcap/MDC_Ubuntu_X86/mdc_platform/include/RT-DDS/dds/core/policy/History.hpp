/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: History.hpp
 */

#ifndef DDS_CORE_POLICY_HISTORY_HPP
#define DDS_CORE_POLICY_HISTORY_HPP

#include <RT-DDS/dds/core/policy/HistoryKind.hpp>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Specifies how much historical data a dds::pub::DataWriter and a
 * dds::sub::DataReader can store.
 */
class History {
public:
    /**
     * @brief Creates a policy that keeps the last sample only.
     */
    History() = default;

    ~History() = default;

    static History KeepLast(int32_t depth) noexcept
    {
        return History(dds::core::policy::HistoryKind::KEEP_LAST, depth);
    }

    static History KeepAll(int32_t depth) noexcept
    {
        return History(dds::core::policy::HistoryKind::KEEP_ALL, depth);
    }

    /**
     * @brief Sets the history kind.
     *
     * Specifies the kind of history to be kept.
     * **[default]** dds::core::policy::HistoryKind::KEEP_LAST
     */
    void Kind(dds::core::policy::HistoryKind kind) noexcept
    {
        kind_ = kind;
    }

    /**
     * @brief Gets the history kind.
     */
    dds::core::policy::HistoryKind Kind() const noexcept
    {
        return kind_;
    }

    /**
     * @brief Sets the history depth.
     *
     * Specifies the number of samples to be kept, when the kind is
     * dds::core::policy::HistoryKind::KEEP_LAST
     */
    void Depth(int32_t depth) noexcept
    {
        depth_ = depth;
    }

    /**
     * @brief Gets the history depth.
     */
    int32_t Depth() const noexcept
    {
        return depth_;
    }

private:
    /**
     * @brief Creates a policy with a specific history kind and optionally a history depth.
     */
    explicit History(
        dds::core::policy::HistoryKind kind, int32_t depth) noexcept
        : kind_(kind), depth_(depth)
    {}

    dds::core::policy::HistoryKind kind_{HistoryKind::KEEP_LAST};
    int32_t depth_{1};
};
}
}
}

#endif /* DDS_CORE_POLICY_HISTORY_HPP */

