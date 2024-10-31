/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataWriterListener.hpp
 */

#ifndef DDS_PUB_DATA_WRITER_LISTENER_HPP
#define DDS_PUB_DATA_WRITER_LISTENER_HPP

#include <memory>
#include <RT-DDS/dds/core/status/PublicationMatchedStatus.hpp>
#include <RT-DDS/dds/core/status/OfferedDeadlineMissedStatus.hpp>

namespace dds {
namespace pub {
class AnyDataWriterImpl;
using DataWriterImplPtr = std::shared_ptr<dds::pub::AnyDataWriterImpl>;
/**
 * @brief The Listener to notify status changes for a dds::pub::DataWriter.
 */
class DataWriterListener {
public:
    /**
     * @brief Handles the dds::core::status::PublicationMatchedStatus status.
     * @details This callback is called when the dds::pub::DataWriter has found
     * a dds::sub::DataReader that matches the dds::topic::Topic with compatible
     * QoS, or has ceased to be matched with a dds::sub::DataReader that was
     * previously considered to be matched.
     * @param[out] writer  Locally created dds::pub::DataWriter that triggers the listener callback.
     * @param[out] status  Current publication match status of locally created dds::pub::DataWriter.
     */
    /* AXIVION Next Line AutosarC++19_03-M0.1.8 : this implement is to prevent forcing user to override this callback */
    virtual void OnPublicationMatched(
        DataWriterImplPtr writer,
        const dds::core::status::PublicationMatchedStatus &status)
    {
        static_cast<void>(writer);
        static_cast<void >(status);
    }

    virtual bool OnShmCreated(DataWriterImplPtr writer, const std::string &shmName)
    {
        static_cast<void>(writer);
        static_cast<void>(shmName);
        return true;
    }

    /**
    * @brief Handles the dds::core::status::OnOfferedDeadlineMissedStatus
    * @param reader Locally created dds::sub::DataWriter that triggers the listener callback.
    * @param status Current Offered Deadline Missed Status of locally created dds::sub::DataWriter.
    * @return void
    */
    virtual void OnOfferedDeadlineMissed(
            DataWriterImplPtr writer,
            const dds::core::status::OfferedDeadlineMissedStatus &status)
    {
        static_cast<void>(writer);
        static_cast<void>(status);
    }

    virtual ~DataWriterListener() = default;
};
}
}

#endif /* DDS_PUB_DATA_WRITER_LISTENER_HPP */
