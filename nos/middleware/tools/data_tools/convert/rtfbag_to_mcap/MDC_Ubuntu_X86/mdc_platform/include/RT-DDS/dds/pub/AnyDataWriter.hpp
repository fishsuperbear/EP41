/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: AnyDataWriter.hpp
 */

#ifndef DDS_PUB_ANY_DATA_WRITER_HPP
#define DDS_PUB_ANY_DATA_WRITER_HPP

#include <functional>

#include <RT-DDS/dds/core/Guid.hpp>
#include <RT-DDS/dds/core/Entity.hpp>
#include <RT-DDS/dds/core/status/StatusMask.hpp>
#include <RT-DDS/dds/topic/AnyTopic.hpp>
#include <RT-DDS/dds/pub/DataWriterListener.hpp>
#include <RT-DDS/dds/pub/qos/DataWriterQos.hpp>
#include <RT-DDS/dds/pub/WriteParams.hpp>
#include <RT-DDS/dds/type/KeyedType.hpp>
#include <RT-DDS/dds/core/InstanceHandle.hpp>
#include <RT-DDS/dds/core/Time.hpp>

#include <RT-DDS/dds/cdr/SerializationCDR.hpp>
#include <RT-DDS/dds/core/status/PublicationMatchedStatus.hpp>
#include <dp_adapter.h>

namespace dds {
namespace pub {
class Publisher;

class AnyDataWriterImpl;

using DataWriterPtr = std::shared_ptr<AnyDataWriterImpl>;

/**
 * @brief Parent class of dds::pub::DataWriter without template type T.
 */
class AnyDataWriter : public dds::core::Entity {
public:
    virtual ~AnyDataWriter(void) = default;

    /**
     * @brief Sets the DataWriter listener.
     * @param[in] listener The DataWriterListener to set.
     * @param[in] mask Changes of communication status to be invoked on the listener.
     * @req{AR-iAOS-RCS-DDS-04108,
     * DataWriter<T> shall support setting of Listener.,
     * QM,
     * DR-iAOS-RCS-DDS-00092
     * }
     */
    void Listener(
        dds::pub::DataWriterListener *listener,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All());

    /**
     * @brief Get the PublicationMatchedStatus.
     * @details This also resets the status so that it is no longer considered changed.
     * @return dds::core::status::PublicationMatchedStatus.
     * @req{AR-iAOS-RCS-DDS-04106,
     * DataWriter<T> shall support getting PublicationMatchedStatus.,
     * QM,
     * DR-iAOS-RCS-DDS-00002
     * }
     */
    dds::core::status::PublicationMatchedStatus PublicationMatchedStatus(void);

    dds::core::status::OfferedDeadlineMissedStatus OfferedDeadlineMissedStatus(void);

    /**
     * @brief Get the Guid.
     * @return dds::core::Guid
     * @req{AR-iAOS-RCS-DDS-04109,
     * DataWriter<T> shall support getting Guid.,
     * QM,
     * DR-iAOS-RCS-DDS-00037
     * }
     */
    dds::core::Guid Guid(void);

    const dds::pub::qos::DataWriterQos& GetQos(void) const;

    core::ReturnCode SetRelatedEntity(const core::Guid &guid);

    const std::string &GetStatString(void);

    /**
     * @brief Waiting for acknowledgments from matched DataReader entities.
     * @details This operation is intended to be used only if the DataWriter
     * has RELIABILITY QoS kind set to RELIABLE. Otherwise the operation will
     * return immediately with dds::core::ReturnCode::OK. The operation
     * WaitForAcknowledgments blocks the calling thread until either all data
     * written by the DataWriter is acknowledged by all matched DataReader
     * entities that have RELIABILITY QoS kind RELIABLE, or else the duration
     * specified by the maxWait parameter elapses, whichever happens first.
     * @param maxWait Duration specified to wait.
     * @return dds::core::ReturnCode::OK All the samples written have been
     * acknowledged by all reliable matched data readers.
     * @return dds::core::ReturnCode::TIMEOUT maxWait elapsed before all the
     * data was acknowledged.
     * @return dds::core::ReturnCode::BAD_PARAMETER The input parameter is not
     * within the allowed range.
     * @return dds::core::ReturnCode::NOT_ENABLED The writer has not been
     * enabled.
     * @return dds::core::ReturnCode::ERROR An internal error has occurred.
     */
    dds::core::ReturnCode WaitForAcknowledgments(dds::core::Duration maxWait);

protected:
    explicit AnyDataWriter(std::shared_ptr<AnyDataWriterImpl> impl) noexcept;

    explicit AnyDataWriter(
        const dds::domain::DomainParticipant &participant,
        const dds::topic::AnyTopic &topic,
        const dds::pub::qos::DataWriterQos &qos,
        dds::pub::DataWriterListener *listener,
        dds::core::status::StatusMask mask) noexcept;

    explicit AnyDataWriter(
        const dds::pub::Publisher &publisher,
        const dds::topic::AnyTopic &topic,
        const dds::pub::qos::DataWriterQos &qos,
        dds::pub::DataWriterListener *listener,
        dds::core::status::StatusMask mask) noexcept;

    AllocateResult AllocateOctetsImpl(
        const std::function<bool(cdr::SerializePayload&)>& dataSerializer, std::size_t dataSize);

    dds::core::ReturnCode WriteZeroCpyDataImpl(const dds::pub::WriteParams& writeParams);

    dds::core::ReturnCode DeallocateZeroCpyDataImpl(ZeroCopyHandle handle);

    dds::core::ReturnCode WriteNormalDataImpl(
        const std::function<bool(cdr::SerializePayload&)>& dataSerializer,
        std::size_t dataSize,
        const dds::pub::WriteParams& writeParams);

    dds::core::ReturnCode WriteMbufDataImpl(Mbuf* pMbuf, const dds::pub::WriteParams& writeParams);

    dds::core::ReturnCode ExpediteWriteMbufDataImpl(Mbuf* pMbuf) const;

    dds::core::InstanceHandle RegisterInstanceImpl(const dds::type::KeyedType& sample, dds::core::Time timeStamp);

    core::ReturnCode UnregisterInstanceImpl(const dds::core::InstanceHandle& handle, dds::core::Time timeStamp);

    core::ReturnCode DisposeInstanceImpl(const dds::core::InstanceHandle& handle, dds::core::Time timeStamp);

    core::ReturnCode GetKeyValueImpl(dds::type::KeyedType& keyHolder, const dds::core::InstanceHandle& handle);

    dds::core::InstanceHandle LookUpInstanceImpl(const dds::type::KeyedType& keyHolder);

    void SetIsKeyed(bool isKeyed) noexcept;

    void SetIsRawData(bool isRawData) noexcept;

    bool isExpediteWriter_ {false};
private:
    std::shared_ptr<AnyDataWriterImpl> impl_;
};
}
}

#endif /* DDS_PUB_ANY_DATA_WRITER_HPP */

