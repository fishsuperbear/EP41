/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: AnyDataReader.hpp
 */

#ifndef DDS_SUB_ANY_DATA_READER_HPP
#define DDS_SUB_ANY_DATA_READER_HPP

#include <functional>
#include <map>

#include <RT-DDS/dds/core/Guid.hpp>
#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/topic/AnyTopic.hpp>
#include <RT-DDS/dds/sub/qos/DataReaderQos.hpp>
#include <RT-DDS/dds/sub/DataReaderListener.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>
#include <RT-DDS/dds/sub/Sample.hpp>
#include <RT-DDS/dds/core/MbufPtr.hpp>
#include <RT-DDS/dds/core/InstanceHandle.hpp>
#include <RT-DDS/dds/cdr/SizeCounterCDR.hpp>
#include <RT-DDS/dds/type/KeyedType.hpp>
#include <RT-DDS/dds/core/Types.hpp>
#include <RT-DDS/dds/sub/SampleBuilder.hpp>

namespace dds {
namespace sub {
class Subscriber;

class DataReaderImpl;

/**
 * @brief Parent class of dds::sub::AnyDataReader without template type T.
 */
class AnyDataReader : public dds::core::Entity {
public:
    ~AnyDataReader(void) override = default;

    /**
     * @brief Sets the DataReader listener.
     * @param[in] listener The DataReaderListener to set.
     * @param[in] mask Changes of communication status to be invoked on the listener.
     */
    void Listener(
        dds::sub::DataReaderListener *listener,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All());

    /**
     * @brief Get the SubscriptionMatchedStatus.
     * @details This also resets the status so that it is no longer considered changed.
     * @return dds::core::status::SubscriptionMatchedStatus.
     */
    dds::core::status::SubscriptionMatchedStatus SubscriptionMatchedStatus(void);

    dds::core::status::RequestedDeadlineMissedStatus RequestedDeadlineMissedStatus(void);

    /**
    * @brief Get the Guid.
    * @return dds::core::Guid
    * @req{AR-iAOS-RCS-DDS-05507,
    * DataReader<T> shall support getting Guid.,
    * QM,
    * DR-iAOS-RCS-DDS-00138
    * }
    */
    dds::core::Guid Guid(void) const;

    core::ReturnCode SetRelatedEntity(const core::Guid &guid);

    /**
     * @brief This is a temporary function to check if there are any packs in the
     * DataReader can be taken
     * @return true means there are samples can be taken
     * @note Only for current check, there could be some concurrent situations
     */
    bool HavePackToReadTmp();

    /**
    * @ingroup DataReader
    * @brief Get Reader history cache status
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] void
    * ...
    * @return dds::core::CacheStatus
    * ...
    * @req{AR-iAOS-RCS-DDS-AR20220509590676, AR20220509590745
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * DR-iAOS-RCS-DDS-XXXXX
    * }
    */
    dds::core::CacheStatus GetDRCacheStatus();

    core::ReturnCode SetLatencyTimeOut(uint32_t timeOut) noexcept;

    /**
     * @ingroup DataReader
     * @brief ask dds to store the latest even after callback ends
     * @return indication of whether this is successful
     */
    bool UpdateExpediteTakeCache(SampleBuilder&& builder);

protected:
    void ReturnLoanBySingleHandleImpl(ZeroCopyHandle handle);

    using DeserHandle = std::function<bool(const dds::cdr::SerializePayload&, dds::sub::SampleInfo&)>;

    using KeyedDeserHandle = std::function<bool(
        const dds::cdr::SerializePayload&, const dds::sub::SampleInfo&,
        const std::function<bool(const type::KeyedType&)>&)>;

    explicit AnyDataReader(
        DataReaderImplPtr impl) noexcept;

    explicit AnyDataReader(
        const dds::domain::DomainParticipant &participant,
        const dds::topic::AnyTopic &topic,
        dds::sub::qos::DataReaderQos qos,
        dds::sub::DataReaderListener *listener,
        dds::core::status::StatusMask mask) noexcept;

    explicit AnyDataReader(
        const dds::sub::Subscriber &sub,
        const dds::topic::AnyTopic &topic,
        dds::sub::qos::DataReaderQos qos,
        dds::sub::DataReaderListener *listener,
        dds::core::status::StatusMask mask) noexcept;

    explicit AnyDataReader(
        const dds::domain::DomainParticipant &participant,
        dds::sub::DataReaderListener *listener,
        dds::core::status::StatusMask mask);

    dds::core::ReturnCode TakeSamples(uint32_t maxSample, bool loan, const DeserHandle& deserHandle);

    dds::core::ReturnCode TakeSamplesWithKey(
        uint32_t maxSample, const KeyedDeserHandle& deserHandle, const core::InstanceHandle& targetIns);

    /* rawdata needn't returnload, so needn't handle_. */
    std::vector<dds::sub::Sample<dds::core::MbufPtr>> LoanRawDataPayload(uint32_t maxSample);

    void ReturnLoanInGroup(const std::vector<uint64_t>& handles);

    const dds::sub::qos::DataReaderQos& GetDataReaderQos(void) const;

    const std::string &GetStatStringImpl(void);

    /**
    * @ingroup DataReader
    * @brief Get Reader statistics
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] resetControl a 32bits bitset to control reset of each StatisticKind.
    * ...
    * @return std::map<dds::core::StatisticKind, uint64_t>
    * ...
    * @req{AR-iAOS-RCS-DDS-AR20220509590676, AR20220509590745
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * DR-iAOS-RCS-DDS-XXXXX
    * }
    */
    const std::map<dds::core::StatisticKind, uint64_t> GetStatImpl(std::bitset<16U> resetControl = 0XE0U);

    core::ReturnCode GetKeyValueImpl(dds::type::KeyedType& keyHolder, const dds::core::InstanceHandle& handle);

    dds::core::InstanceHandle LookUpInstanceImpl(const dds::type::KeyedType& keyHolder);

    void SetIsKeyed(bool isKeyed) noexcept;

    static bool CheckTakeMaxSampleInput(std::int32_t nCount);

    /**
     * @brief register a deserialize function to be used by OnDataProcess
     * @param function a serialization closure that formed in previous step
     * @return void
     */
    void RegisterDeserializeFunction(const std::function<bool(SampleBase&, cdr::SerializePayload)>& function) noexcept;

    /**
     * @brief register a deserialize function to be used by OnDataProcess
     * @param isRawData boolean value indicating if this is a rawData reader
     * @return void
     */
    void SetIsRawData(bool isRawData) noexcept;

    std::shared_ptr<DataReaderImpl> impl_;
};
}
}

#endif /* DDS_SUB_ANY_DATA_READER_HPP */

