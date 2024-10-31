/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Data statistician header
 * Create: 2021-06-10
 */
#ifndef VRTF_VCC_UTILS_DATA_STATISTICIAN_H
#define VRTF_VCC_UTILS_DATA_STATISTICIAN_H

#include <mutex>

#include "vrtf/vcc/utils/stats/stats.h"

namespace rbs {
namespace plog {
namespace stat {
    class InfoHandle;
    class InfoWriter;
}
}
}
namespace vrtf {
namespace vcc {
namespace utils {
namespace stats {
class DataStatistician final {
public:
    using StatisticsTask = std::function<void()>;
    using PeriodType     = std::uint32_t;

    static std::shared_ptr<DataStatistician> &GetInstance();

    DataStatistician();

    ~DataStatistician() = default;

    // no copyable
    DataStatistician(DataStatistician const&) = delete;
    DataStatistician& operator=(DataStatistician const&) = delete;

    bool IsEnable() const noexcept;

    PeriodType Period() const noexcept;

    bool CreatePlogInfoWriter(EntityIdentifier const &entityIdentifier) noexcept;

    void RemovePlogInfoWriter(EntityIdentifier const &entityIdentifier) noexcept;

    bool BeginNewPlogOutput(EntityIdentifier const &entityIdentifier) noexcept;

    bool PlogOutput(EntityIdentifier const &entityIdentifier, std::string const &str) noexcept;

    bool FinishCurrentPlogOutput(EntityIdentifier const &entityIdentifier) noexcept;

private:
    using InfoHandlePtr = std::shared_ptr<rbs::plog::stat::InfoHandle>;
    using InfoWriterPtr = std::shared_ptr<rbs::plog::stat::InfoWriter>;

    InfoWriterPtr FindInfoWriter(EntityIdentifier const &entityIdentifier) noexcept;

    static std::string GenInfoHandleName() noexcept;

    std::shared_ptr<ara::godel::common::log::Log> logInstance_ {nullptr};
    bool enable_ {false};
    std::uint32_t period_ {0U};
    InfoHandlePtr infoHandle_ {nullptr};
    std::map<EntityIdentifier, InfoWriterPtr> mapInfoWriter_ {};
    std::mutex infoWriterMutex_;
};
} // namespace vrtf
} // namespace vcc
} // namespace utils
} // namespace stats
#endif
