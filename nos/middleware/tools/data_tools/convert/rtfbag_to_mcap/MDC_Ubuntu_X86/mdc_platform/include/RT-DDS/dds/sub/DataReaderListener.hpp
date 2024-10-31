/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataReaderListener.hpp
 */

#ifndef DDS_SUB_DATA_READER_LISTENER_HPP
#define DDS_SUB_DATA_READER_LISTENER_HPP

#include <memory>
#include <RT-DDS/dds/core/status/SubscriptionMatchedStatus.hpp>
#include <RT-DDS/dds/core/status/SampleLostStatus.hpp>
#include <RT-DDS/dds/sub/SampleBuilder.hpp>
#include <RT-DDS/dds/core/status/RequestedDeadlineMissedStatus.hpp>
#include <RT-DDS/dds/core/status/SampleTimeOutStatus.hpp>

namespace dds {
namespace sub {
class DataReaderImpl;
using DataReaderImplPtr = std::shared_ptr<dds::sub::DataReaderImpl>;
/**
 * @brief The Listener to notify status changes for a dds::sub::DataReader.
 */
class DataReaderListener {
public:
    /**
     * @brief Called when one or more new data samples have been received.
     * @param[out] reader Locally created dds::sub::DataReader that triggers the listener callback.
     */
    /* AXIVION Next Line AutosarC++19_03-M0.1.8 : this implement is to prevent forcing user to override this callback */
    virtual void OnDataAvailable(
        DataReaderImplPtr reader)
    {
        static_cast<void>(reader);
    }

    /**
     * @brief Called when one new data sample is ready for process.
     * @param[out] reader Locally created dds::sub::DataReader that triggers the listener callback.
     */
    /* AXIVION Next Line AutosarC++19_03-M0.1.8 : this implement is to prevent forcing user to override this callback */
    virtual void OnDataProcess(
        const DataReaderImplPtr& reader, SampleBuilder&& sampleBuilder)
    {
        /// should write to `T sample` section of a recvBuffer
        /// and call `sampleBuilder.BuildSample(sample)` to direct deserialize to write
        static_cast<void>(reader);
        static_cast<void>(sampleBuilder);
    }

    /**
     * @ingroup DataReaderListener
     * @brief The OnSampleLostListener callback function
     * @param[in] reader DataReaderImplPtr
     * @param[in] status store the samplelost statistics
     * @param[out] None
     * @return void
     * @req {AR20220610482076}
     */
    virtual void OnSampleLost(
        DataReaderImplPtr reader,
        const dds::core::status::SampleLostStatus &status)
    {
        static_cast<void>(reader);
        static_cast<void>(status);
    }

    /**
     * @brief Handles the dds::core::status::SubscriptionMatchedStatus status.
     * @details This callback is called when the dds::sub::DataReader has found
     * a dds::pub::DataWriter that matches the dds::topic::Topic with compatible
     * QoS, or has ceased to be matched with a dds::pub::DataWriter that was
     * previously considered to be matched.
     * @param[out] reader  Locally created dds::sub::DataReader that triggers the listener callback.
     * @param[out] status  Current subscription match status of locally created dds::sub::DataReader.
     */
    /* AXIVION Next Line AutosarC++19_03-M0.1.8 : this implement is to prevent forcing user to override this callback */
    virtual void OnSubscriptionMatched(
        DataReaderImplPtr reader,
        const dds::core::status::SubscriptionMatchedStatus &status)
    {
        static_cast<void>(reader);
        static_cast<void>(status);
    }

    /**
     * @brief Handles the dds::core::status::RequestedDeadlineMissedStatus
     * @param reader Locally created dds::sub::DataReader that triggers the listener callback.
     * @param status Current OnRequestedDeadlineMissedstatus of locally created dds::sub::DataReader.
     * @return void
     */
    virtual void OnRequestedDeadlineMissed(
        DataReaderImplPtr reader,
        const dds::core::status::RequestedDeadlineMissedStatus &status)
    {
        static_cast<void>(reader);
        static_cast<void>(status);
    }

    /**
     * @ingroup DataReaderListener
     * @brief The OnSampleTimeOutListener callback function
     * @param[in] reader DataReaderImplPtr
     * @param[in] status store the samplelTimeOut statistics
     * @param[out] None
     * @return void
     */
    virtual void OnSampleTimeOut(
        DataReaderImplPtr reader,
        const dds::core::status::SampleTimeOutStatus &status)
    {
        static_cast<void>(reader);
        static_cast<void>(status);
    }

    virtual ~DataReaderListener() = default;
};
}
}

#endif /* DDS_SUB_DATA_READER_LISTENER_HPP */

