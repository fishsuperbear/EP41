/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef RT_DDS_SAMPLEBUILDER_HPP
#define RT_DDS_SAMPLEBUILDER_HPP

#include <functional>
#include <RT-DDS/dds/sub/SampleBase.hpp>
#include <RT-DDS/dds/sub/SampleInfo.hpp>
#include <RT-DDS/dds/cdr/SerializationCDR.hpp>

namespace dds {
namespace sub {
/**
 * @brief This class encapsulate the WriteClosure and SampleInfo associated with DDS samples.
 */
class SampleBuilder {
public:

    SampleBuilder(const SampleBuilder& builder) = delete;
    SampleBuilder& operator=(const SampleBuilder& builder) = delete;

    SampleBuilder(SampleBuilder&& builder) noexcept;

    SampleBuilder& operator=(SampleBuilder&& builder) noexcept;

    SampleBuilder() noexcept;

    /**
     * @brief function to be used by the user to deserialize data to a given location
     * @param base location to be write to
     * @return whether build is successful
     */
    bool BuildSample(SampleBase& base) noexcept;

    /**
     * @brief Get the SampleInfo associated with this Sample.
     * @return const reference of SampleInfo.
     */
    const SampleInfo& GetSampleInfo() const noexcept;

private:
    SampleBuilder(const std::function<bool(SampleBase&, cdr::SerializePayload)>& deserializationClosure,
                  dds::cdr::SerializePayload payload,
                  const SampleInfo& info) noexcept;

    const std::function<bool(SampleBase&, cdr::SerializePayload)>* writeClosure_;
    cdr::SerializePayload payload_{};
    SampleInfo info_;
    bool used_;

    friend class AnyDataReaderImpl;
};
}
}

#endif // RT_DDS_SAMPLEBUILDER_HPP
