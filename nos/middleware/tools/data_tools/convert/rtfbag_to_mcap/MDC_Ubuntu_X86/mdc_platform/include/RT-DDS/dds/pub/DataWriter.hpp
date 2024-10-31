/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataWriter.hpp
 */

#ifndef DDS_PUB_DATA_WRITER_HPP
#define DDS_PUB_DATA_WRITER_HPP

#include <RT-DDS/dds/topic/Topic.hpp>
#include <RT-DDS/dds/pub/AnyDataWriter.hpp>
#include <RT-DDS/dds/pub/DataWriterListener.hpp>
#include <RT-DDS/dds/pub/WriteParams.hpp>
#include <RT-DDS/dds/core/MbufPtr.hpp>
#include <RT-DDS/dds/type/KeyedType.hpp>
#include <RT-DDS/dds/core/InstanceHandle.hpp>

namespace dds {
namespace pub {
/**
 * @brief Allows an application to publish data for a dds::topic::Topic.
 * @tparam T User-defined sample type.
 */
template<typename T>
class DataWriter : public AnyDataWriter {
public:
    /**
     * @brief Creates a DataWriter with std::shared_ptr<AnyDataWriterImpl>
     * @param[in] impl Internal implementation of DataWriter.
     * @req{AR-iAOS-RCS-DDS-04101,
     * DataWriter<T> shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00021
     * }
     */
    explicit DataWriter(std::shared_ptr<AnyDataWriterImpl> impl) noexcept
        : AnyDataWriter(std::move(impl))
    {}

    /**
     * @brief Creates a DataWriter with QoS and listener.
     * @param[in] participant The DomainParticipant that this DataWriter belongs.
     * @param[in] topic  The Topic associated with this DataWriter.
     * @param[in] qos    QoS to be used for creating the new DataWriter.
     * @param[in] listener  The DataWriter listener.
     * @param[in] mask	Changes of communication status to be invoked on the listener.
     * @req{AR-iAOS-RCS-DDS-04101,
     * DataWriter<T> shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00021
     * }
     */
    explicit DataWriter(
        const dds::domain::DomainParticipant &participant,
        const dds::topic::Topic<T> &topic,
        const dds::pub::qos::DataWriterQos &qos = dds::pub::qos::DataWriterQos(),
        DataWriterListener *listener = nullptr,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All()) noexcept
        : AnyDataWriter(participant, topic, qos, listener, mask)
    {
        SetIsKeyed(std::is_base_of<dds::type::KeyedType, T>::value);
        SetIsRawData(std::is_same<dds::core::MbufPtr, T>::value);
    }

    /**
     * @brief Creates a DataWriter with QoS and listener.
     * @param[in] publisher The Publisher that this DataWriter belongs.
     * @param[in] topic  The Topic associated with this DataWriter.
     * @param[in] qos    QoS to be used for creating the new Datawriter.
     * @param[in] listener  The DataWriter listener.
     * @param[in] mask	Changes of communication status to be invoked on the listener.
     * @req{AR-iAOS-RCS-DDS-04101,
     * DataWriter<T> shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00021
     * }
     */
    explicit DataWriter(
        const dds::pub::Publisher &publisher,
        const dds::topic::Topic<T> &topic,
        const dds::pub::qos::DataWriterQos &qos = dds::pub::qos::DataWriterQos(),
        DataWriterListener *listener = nullptr,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All()) noexcept
        : AnyDataWriter(publisher, topic, qos, listener, mask)
    {
        SetIsKeyed(std::is_base_of<dds::type::KeyedType, T>::value);
        SetIsRawData(std::is_same<dds::core::MbufPtr, T>::value);
    }

    /**
     * @brief Default destructor.
     * @req{AR-iAOS-RCS-DDS-04102,
     * DataWriter<T> shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00021
     * }
     */
    ~DataWriter(void) override = default;

    /** the following are non-key type */
    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        dds::core::InstanceHandle>::type
    RegisterInstance(const T& instanceData, dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        static_cast<void>(instanceData);
        static_cast<void>(timeStamp);
        return dds::core::InstanceHandle::Nil();
    }

    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    UnregisterInstance(const dds::core::InstanceHandle& handle,
                       dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        static_cast<void>(handle);
        static_cast<void>(timeStamp);
        return core::ReturnCode::PRECONDITION_NOT_MET;
    }

    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    DisposeInstance(const dds::core::InstanceHandle& handle,
                    dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        static_cast<void>(handle);
        static_cast<void>(timeStamp);
        return core::ReturnCode::PRECONDITION_NOT_MET;
    }

    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    GetKeyValue(T& keyHolder, const dds::core::InstanceHandle& handle) noexcept
    {
        static_cast<void>(keyHolder);
        static_cast<void>(handle);
        return core::ReturnCode::PRECONDITION_NOT_MET;
    }

    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        dds::core::InstanceHandle>::type
    LookUpInstance(const T& keyHolder)
    {
        static_cast<void>(keyHolder);
        return dds::core::InstanceHandle::Nil();
    }

    /** the following are for keyed type */

    /**
     * @brief register a instance to the DataWriter
     * @param instanceData a demo data to tell the key values of the instance
     * @param timeStamp the time point of the registration, which can be calculated by DCPS if input infinite
     * @return a handle to represent the instance, if failed, return InstanceHandle::Nil
     * @note double register will return the same InstanceHandle
     * @req{AR-iAOS-RCS-DDS-01001,
     * DCPS shall be able to manage keyed data and instances,
     * QM,
     * DR-iAOS3-RCS-DDS-00152, DR-iAOS3-RCS-DDS-00153
     * }
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        dds::core::InstanceHandle>::type
    RegisterInstance(const T& instanceData, dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        return RegisterInstanceImpl(instanceData, timeStamp);
    }

    /**
     * @brief Unregister a instance, which must be registered, which means the writer has noting to say on this instance
     * @note when all writers unregister the same instance, reader will get a sample with instance
     *       state: NotAliveNoWriters
     * @param handle the handle of the instance, which is made from RegisterInstance
     * @param timeStamp the time point of the unregister, which can be calculated by DCPS if input infinite
     * @return standard return code, which can be converted to string by ReturnCodeToString
     * @req{AR-iAOS-RCS-DDS-01001,
     * DCPS shall be able to manage keyed data and instances,
     * QM,
     * DR-iAOS3-RCS-DDS-00152, DR-iAOS3-RCS-DDS-00153
     * }
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    UnregisterInstance(const dds::core::InstanceHandle& handle,
                       dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        return UnregisterInstanceImpl(handle, timeStamp);
    }

    /**
     * @brief Dispose a instance, which must be registered, which means the writer active shutdown the instance
     * @param handle the handle of the instance, which is made from RegisterInstance
     * @param timeStamp the time point of the dispose, which can be calculated by DCPS if input infinite
     * @return standard return code, which can be converted to string by ReturnCodeToString
     * @req{AR-iAOS-RCS-DDS-01001,
     * DCPS shall be able to manage keyed data and instances,
     * QM,
     * DR-iAOS3-RCS-DDS-00152, DR-iAOS3-RCS-DDS-00153
     * }
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    DisposeInstance(const dds::core::InstanceHandle& handle,
                    dds::core::Time timeStamp = dds::core::Time::Invalid())
    {
        return DisposeInstanceImpl(handle, timeStamp);
    }

    /**
     * @brief Get the key value of a instance
     * @param keyHolder a sample to store the key value of the instance
     * @param handle the handle of the instance, which is made from RegisterInstance
     * @return standard return code, which can be converted to string by ReturnCodeToString
     * @req{AR-iAOS-RCS-DDS-01001,
     * DCPS shall be able to manage keyed data and instances,
     * QM,
     * DR-iAOS3-RCS-DDS-00152, DR-iAOS3-RCS-DDS-00153
     * }
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        core::ReturnCode>::type
    GetKeyValue(T& keyHolder, const dds::core::InstanceHandle& handle)
    {
        return GetKeyValueImpl(keyHolder, handle);
    }

    /**
     * @brief look up the instance handle of a given key type
     * @param keyHolder a T sample, whose key values are used to find the target instance
     * @return the InstanceHandle of the instance if found, else will return InstanceHandle::Nil
     * @req{AR-iAOS-RCS-DDS-01001,
     * DCPS shall be able to manage keyed data and instances,
     * QM,
     * DR-iAOS3-RCS-DDS-00152, DR-iAOS3-RCS-DDS-00153
     * }
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        dds::core::InstanceHandle>::type
    LookUpInstance(const T& keyHolder)
    {
        return LookUpInstanceImpl(keyHolder);
    }

    /**
     * @brief Modifies the value of a data instance.
     * @param instanceData The data sample to write.
     * @param writeParams  data instance handle and reader guid to be sent.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::NOT_ENABLED
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     * @req{AR-iAOS-RCS-DDS-04107, AR-iAOS-RCS-DDS-04112
     * DataWriter<T> shall support writing of Sample<T>,
     * DataWriter<T> shall support writing with Reader Guid.
     * QM,
     * DR-iAOS-RCS-DDS-00037, DR-iAOS-RCS-DDS-00137
     * }
     */
    dds::core::ReturnCode Write(const T &instanceData,
                                const dds::pub::WriteParams& writeParams = dds::pub::WriteParams{})
    {
        dds::cdr::SerializationCDR<T> serializer{const_cast<T &>(instanceData)};
        std::size_t dataSize{serializer.GetSize()};

        std::function<bool(dds::cdr::SerializePayload&)> f = [&serializer](const dds::cdr::SerializePayload& payload) {
            return serializer.Serialize(payload);
        };

        return WriteNormalDataImpl(f, dataSize, writeParams);
    }

    /**
     * @brief Allocate memory for a data instance.
     * @details This operation allocates memory internally for a data instance
     * and modifies dds::core::Octets field for zero-copy transfer.
     * @param[in] instanceData The data sample to allocate memory.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::NOT_ENABLED
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     * @req{AR-iAOS-RCS-DDS-04110,
     * DataWriter<T> shall support allocating memory for sample.,
     * QM,
     * DR-iAOS-RCS-DDS-00037
     * }
     */
    AllocateResult AllocateOctets(T &instanceData)
    {
        dds::cdr::SerializationCDR<T> serializer{const_cast<T &>(instanceData)};
        std::size_t dataSize{serializer.GetSize()};

        std::function<bool(dds::cdr::SerializePayload&)> f = [&serializer](const dds::cdr::SerializePayload& payload) {
            return serializer.Serialize(payload);
        };

        return AllocateOctetsImpl(f, dataSize);
    }

    /**
     * @brief Write out zero copy sample allocated from AllocateOctets.
     * @param writeParams data instance handle and reader guid to be sent.
     * @return dds::core::ReturnCode
     * @retval dds::core::ReturnCode::OK
     * @retval dds::core::ReturnCode::ERROR
     * @retval dds::core::ReturnCode::NOT_ENABLED
     * @retval dds::core::ReturnCode::ALREADY_DELETED
     * @req{AR-iAOS-RCS-DDS-04111, AR-iAOS-RCS-DDS-04112
     * DataWriter<T> shall support writing out sample allocated before,
     * DataWriter<T> shall support writing with Reader Guid.
     * QM,
     * DR-iAOS-RCS-DDS-00037, DR-iAOS-RCS-DDS-00137
     * }
     */
    dds::core::ReturnCode WriteZeroCpyData(const dds::pub::WriteParams& writeParams)
    {
        return AnyDataWriter::WriteZeroCpyDataImpl(writeParams);
    }

    /**
     * @brief Deallocate the sample data allocated from AllocateOctets without writing.
     * @param handle data instance handle and reader guid to be sent.
     * @return dds::core::ReturnCode
     */
    dds::core::ReturnCode DeallocateZeroCpyData(ZeroCopyHandle handle)
    {
        return AnyDataWriter::DeallocateZeroCpyDataImpl(handle);
    }

    /**
     * @brief Expedite send sample data
     * @param instanceData data instance to be sent
     * @return dds::core::ReturnCode
     */
    dds::core::ReturnCode ExpediteWrite(const T &instanceData) const noexcept {
        static_cast<void>(instanceData);
        return dds::core::ReturnCode::UNSUPPORTED;  // need manual specialization support for different types
    }
};

/**
 * @brief Alloc change for rawdata mbuf, and send it.
 * @param instanceData the mbuf of rawdata.
 * @param writeParams  data instance handle and reader guid to be sent.
 * @return dds::core::ReturnCode
 * @retval dds::core::ReturnCode::OK
 * @retval dds::core::ReturnCode::ERROR
 * @retval dds::core::ReturnCode::NOT_ENABLED
 * @retval dds::core::ReturnCode::ALREADY_DELETED
 * @req{AR-iAOS-RCS-DDS-12501
 * DataWriter<T> shall support writing of Sample<T>,
 * QM,
 * DR-iAOS-RCS-DDS-00160
 * }
 */
template<>
dds::core::ReturnCode DataWriter<dds::core::MbufPtr>::Write(const dds::core::MbufPtr &instanceData,
                                                            const dds::pub::WriteParams& writeParams);

template<>
dds::core::ReturnCode DataWriter<dds::core::MbufPtr>::ExpediteWrite(const dds::core::MbufPtr& instanceData)
    const noexcept;

}
}

#endif /* DDS_PUB_DATA_WRITER_HPP */

