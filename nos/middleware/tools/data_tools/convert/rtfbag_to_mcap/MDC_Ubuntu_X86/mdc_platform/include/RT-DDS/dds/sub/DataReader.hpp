/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description: DataReader.hpp
 */

#ifndef DDS_SUB_DATA_READER_HPP
#define DDS_SUB_DATA_READER_HPP

#include <RT-DDS/dds/core/status/StatusMask.hpp>
#include <RT-DDS/dds/domain/DomainParticipant.hpp>
#include <RT-DDS/dds/topic/Topic.hpp>
#include <RT-DDS/dds/topic/ParticipantBuiltinTopicData.hpp>
#include <RT-DDS/dds/sub/Subscriber.hpp>
#include <RT-DDS/dds/sub/qos/DataReaderQos.hpp>
#include <RT-DDS/dds/sub/DataReaderListener.hpp>
#include <RT-DDS/dds/sub/AnyDataReader.hpp>
#include <RT-DDS/dds/sub/TakeParams.hpp>
#include <RT-DDS/dds/cdr/SerializationCDR.hpp>
#include <RT-DDS/dds/sub/SampleBase.hpp>
#include <RT-DDS/dds/sub/SampleBuilder.hpp>

namespace dds {
namespace sub {
class Subscriber;

/**
 * @brief Allows the application to: (1) declare the data it wishes to receive
 * (i.e. make a subscription) and (2) access the data received.
 * @tparam T User-defined sample type.
 */
template<typename T>
class DataReader : public AnyDataReader {
public:
    explicit DataReader(DataReaderImplPtr impl) noexcept
        : AnyDataReader(impl)
    {}

    /**
     * @brief Creates a User-defined DataReader with QoS and listener.
     * @param[in] participant The DomainParticipant that this DataReader belongs.
     * @param[in] topic  The Topic associated with this DataReader.
     * @param[in] qos    QoS to be used for creating the new DataReader.
     * @param[in] listener  The DataReader listener.
     * @param[in] mask	Changes of communication status to be invoked on the listener.
     * @req{AR-iAOS-RCS-DDS-05501,
     * DataReader<T> shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00026
     * }
     */
    explicit DataReader(
        dds::domain::DomainParticipant participant,
        dds::topic::Topic<T> topic,
        dds::sub::qos::DataReaderQos qos = dds::sub::qos::DataReaderQos(),
        dds::sub::DataReaderListener *listener = nullptr,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All())
        : AnyDataReader(participant, topic, qos, listener, mask)
    {
        SetIsKeyed(std::is_base_of<dds::type::KeyedType, T>::value);
        RegisterDeserializeFunction([](SampleBase& base, dds::cdr::SerializePayload payload) {
            return DeserializationClosure(base, payload);
        });
        SetIsRawData(std::is_same<dds::core::MbufPtr, T>::value);
    }

    /**
     * @brief Creates a User-defined DataReader with QoS and listener.
     * @param[in] sub The Subscriber that this DataReader belongs.
     * @param[in] topic  The Topic associated with this DataReader.
     * @param[in] qos    QoS to be used for creating the new DataReader.
     * @param[in] listener  The DataReader listener.
     * @param[in] mask	Changes of communication status to be invoked on the listener.
     * @req{AR-iAOS-RCS-DDS-05501,
     * DataReader<T> shall support creation process.,
     * QM,
     * DR-iAOS-RCS-DDS-00026
     * }
     */
    explicit DataReader(
        dds::sub::Subscriber sub,
        dds::topic::Topic<T> topic,
        dds::sub::qos::DataReaderQos qos = dds::sub::qos::DataReaderQos(),
        dds::sub::DataReaderListener *listener = nullptr,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All())
        : AnyDataReader(sub, topic, qos, listener, mask)
    {
        SetIsKeyed(std::is_base_of<dds::type::KeyedType, T>::value);
        RegisterDeserializeFunction([](SampleBase& base, dds::cdr::SerializePayload payload) {
            return DeserializationClosure(base, payload);
        });
        SetIsRawData(std::is_same<dds::core::MbufPtr, T>::value);
    }

    /**
     * @brief Default Destructor.
     * @req{AR-iAOS-RCS-DDS-05502,
     * DataReader<T> shall support destruction process.,
     * QM,
     * DR-iAOS-RCS-DDS-00026
     * }
     */
    ~DataReader(void) override = default;

    /**
     * @brief Take samples available in the reader cache samples using the based
     * on maxSamples required.
     * @param[in] maxSample  Number of samples to take.
     * @param[in] loan       Indicate whether to loan samples.
     * @return SampleSeq of type T.
     * @req{AR-iAOS-RCS-DDS-05505,
     * DataReader<T> shall support taking Sample<T>.,
     * QM,
     * DR-iAOS-RCS-DDS-00037
     * }
     * @req{AR-iAOS-RCS-DDS-05506,
     * DataReader<T> shall support loan of Sample<T> and return of it.,
     * QM,
     * DR-iAOS-RCS-DDS-00037
     * }
     */
    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        SampleSeq<T>>::type
    Take(int32_t maxSample, bool loan = false)
    {
        SampleSeq<T> samples{};
        if (!CheckTakeMaxSampleInput(maxSample)) {
            return samples;
        }

        samples.reserve(maxSample);
        DeserHandle fHandle = [&samples](const dds::cdr::SerializePayload& payload, const dds::sub::SampleInfo& info) {
            samples.emplace_back();
            samples.back().info_ = info;
            if (dds::cdr::SerializationCDR<T>{samples.back().data_}.Deserialize(payload) == false) {
                samples.pop_back();
                return false;
            } else {
                return true;
            }
        };

        static_cast<void>(TakeSamples(static_cast<uint32_t>(maxSample), loan, fHandle));

        return samples;
    }

    template<class U = T>
    typename std::enable_if<!std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        SampleSeq<T>>::type
    TakeWithParams(const TakeParams& params) noexcept
    {
        return Take(params.GetMaxSample(), true);
    }

    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        SampleSeq<T>>::type
    Take(int32_t maxSample) noexcept
    {
        TakeParams params{};
        params.SetMaxSample(maxSample);
        return TakeWithParams(params);
    }

    /**
     * @brief Take samples according the configs in the TakeParams
     * @param params the configs to Take the sample @see TakeParams
     * @return a sample sequence
     */
    template<class U = T>
    typename std::enable_if<std::is_base_of<dds::type::KeyedType, U>::value && std::is_same<U, T>::value,
        SampleSeq<T>>::type
    TakeWithParams(const TakeParams& params)
    {
        SampleSeq<T> samples{};
        if (!CheckTakeMaxSampleInput(params.GetMaxSample())) {
            return samples;
        }

        samples.reserve(params.GetMaxSample());
        KeyedDeserHandle fHandle = [&samples](
            const dds::cdr::SerializePayload& payload, const dds::sub::SampleInfo& info,
            const std::function<bool(const type::KeyedType&)>& keyTypeProcessor) {
            samples.emplace_back();
            samples.back().info_ = info;
            if (info.Valid()) {
                if (dds::cdr::SerializationCDR<T>{samples.back().data_}.Deserialize(payload) == false) {
                    samples.pop_back();
                    return false;
                }
            }
            /// sometime more processing is needed
            if (keyTypeProcessor && !keyTypeProcessor(samples.back().Data())) {
                samples.pop_back();
                return false;
            }
            return true;
        };

        static_cast<void>(TakeSamplesWithKey(params.GetMaxSample(), fHandle, params.GetInstanceHandle()));

        return samples;
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

    /**
     * @brief Get the key value of a instance
     * @param keyHolder a sample to store the key value of the instance
     * @param handle the handle of the instance, which must be alive and got from the SampleInfo
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
     * @brief Returns the samples to the DataReader.
     * @details This operation tells the dds::sub::DataReader that the
     * application is done accessing the collection of samples.
     * @param[in] samples
     * @return dds::core::ReturnCode
     * @req{AR-iAOS-RCS-DDS-05506,
     * DataReader<T> shall support loan of Sample<T> and return of it.,
     * QM,
     * DR-iAOS-RCS-DDS-00037
     * }
     */
    void ReturnLoan(const SampleSeq<T> &samples)
    {
        const auto& nSample = samples.size();
        std::vector<uint64_t> handles;
        handles.reserve(nSample);
        for (auto& sampleGot : samples) {
            handles.emplace_back(sampleGot.Info().GetZeroCopyHandle());
        }
        AnyDataReader::ReturnLoanInGroup(handles);
    }

    const dds::sub::qos::DataReaderQos& GetQos(void) const
    {
        return AnyDataReader::GetDataReaderQos();
    }

    /**
     * @ingroup DataReader
     * @brief Ask DDS to save latest change
     * @return indication of whether this operation is successful
     */
    bool PrepareForTake(SampleBuilder&& builder)
    {
        return AnyDataReader::UpdateExpediteTakeCache(std::move(builder));
    }

    const std::string &GetStatString()
    {
        return GetStatStringImpl();
    }

    /**
    * @ingroup DataReader
    * @brief Get Reader statistics
    * @par Description
    * 1. FUNCTION_Description
    * ...
    * @param[in] resetControl 16 bits bitset to control reset status
    * ...
    * @return std::map<dds::core::StatisticKind, uint64_t>
    * ...
    * @req{AR-iAOS-RCS-DDS-AR20220509590676, AR20220509590745
    * NAME_OF_THE_RELATED_AR_NUMBER.,
    * DR-iAOS-RCS-DDS-XXXXX
    * }
    */
    const std::map<dds::core::StatisticKind, uint64_t> GetStat(std::bitset<16U> resetControl = 0xE0U)
    {
        return GetStatImpl(resetControl);
    }

    void ReturnLoanBySingleHandle(ZeroCopyHandle handle)
    {
        ReturnLoanBySingleHandleImpl(handle);
    }

private:
    /**
     * @brief create a type aware deserialization closure
     * @param base location where data should be serialized to
     * @param payload data to be serialized
     * @return an indication of whether the deserialization is success
     */
    template<class U = T>
    static
    typename std::enable_if<std::is_base_of<SampleBase, U>::value &&
             !std::is_same<U, dds::core::MbufPtr>::value &&
             std::is_same<U, T>::value,
                            bool>::type
    DeserializationClosure(SampleBase& base, const dds::cdr::SerializePayload& payload) noexcept
    {
        auto& theT = static_cast<T&>(base);

        return dds::cdr::SerializationCDR<T>{theT}.Deserialize(payload);
    }

    /**
     * @brief create a type aware deserialization closure
     * @param base location where data should be serialized to
     * @param payload data to be serialized
     * @return an indication of whether the deserialization is success
     */
    template<class U = T>
    static
    typename std::enable_if<std::is_same<U, dds::core::MbufPtr>::value &&
             std::is_same<U, T>::value,
                            bool>::type
    DeserializationClosure(SampleBase& base, const dds::cdr::SerializePayload& payload) noexcept
    {
        // Still needs more thoughts on the design.
        auto& dst = static_cast<dds::core::MbufPtr&>(base);
        dst.Buffer(reinterpret_cast<Mbuf *>(payload.buffer));
        return true;
    }

    /**
     * @brief create a type aware deserialization closure
     * @param base location where data should be serialized to
     * @param payload data to be serialized
     * @return an indication of whether the deserialization is success
     */
    template<class U = T>
    static
    typename std::enable_if<!std::is_base_of<SampleBase, U>::value &&
             std::is_same<U, T>::value,
                            bool>::type
    DeserializationClosure(SampleBase& base, const dds::cdr::SerializePayload& payload) noexcept
    {
        static_cast<void>(base);
        static_cast<void>(payload);
        return false;
    }
};

template<>
class DataReader<dds::topic::ParticipantBuiltinTopicData> : public AnyDataReader {
public:
    explicit DataReader(DataReaderImplPtr impl);

    explicit DataReader(
        const dds::domain::DomainParticipant &participant,
        dds::sub::DataReaderListener *listener = nullptr,
        dds::core::status::StatusMask mask = dds::core::status::StatusMask::All());

    SampleSeq<dds::topic::ParticipantBuiltinTopicData> Take(int32_t maxSample) const;

    ~DataReader(void) override = default;
};

template<>template<>
SampleSeq<dds::core::MbufPtr> DataReader<dds::core::MbufPtr>::Take(int32_t maxSample, bool loan);
}
}

#endif /* DDS_SUB_DATA_READER_HPP */

