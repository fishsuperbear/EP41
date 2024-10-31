/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: MethodParamDeserializer in vcc
 * Create: 2019-11-19
 */
#ifndef VRTF_VCC_DRIVER_METHOD_PARAMS_DESERLIZE_H
#define VRTF_VCC_DRIVER_METHOD_PARAMS_DESERLIZE_H
#include <type_traits>
#include <tuple>
#include "vrtf/vcc/api/types.h"
#include "vrtf/vcc/serialize/dds_serialize.h"
#include "vrtf/vcc/serialize/someip_serialize.h"
#include "ara/hwcommon/log/log.h"
namespace vrtf {
namespace vcc {
namespace serialize {
/// \brief Method parameters deserialization
template<class... Args>
class ParamsDeserializer {
public:
    /// \brief ArgType<I> is used to get the Ith parameter's type
    template<std::size_t I>
    using ArgType = typename utils::TemplateDeduction::Parameters<I, Args...>::Type;
    ParamsDeserializer(const vrtf::vcc::api::types::MethodMsg& msg,
        vrtf::serialize::SerializationNode const &requestDeserializationNode,
        vrtf::serialize::SerializeConfig const &config = vrtf::serialize::SerializeConfig ())
        : msg_(msg),
          currentPos_(msg.GetPayload()),
          data_(msg.GetPayload()),
          remainingSize_(msg.GetSize()),
          serializeType_(msg.GetSerializeType()),
          serializeConfig_(config),
          currentNodeConfig_(requestDeserializationNode)
    {
        using namespace ara::godel::common;
        logInstance_ = log::Log::GetLog("CM");
    }
    ~ParamsDeserializer() = default;

    /**
     * @brief Get the Ith parameter
     * @details before this method, IndexMethodParameter must have been called
     * @return The Ith parameter
     */
    template<std::size_t I>
    typename std::decay<ArgType<I>>::type GetValue()
    {
        if (serializeType_ == vrtf::serialize::SerializeType::SHM) {
            vrtf::serialize::dds::Deserializer<typename std::decay<ArgType<I>>::type> deserializer(
                posIndex_[I], argsSize_[I], serializeConfig_);
            typename std::decay<ArgType<I>>::type tmp = deserializer.GetValue();
            return tmp;
        } else if (serializeType_ == vrtf::serialize::SerializeType::SOMEIP) {
            vrtf::serialize::someip::Deserializer<typename std::decay<ArgType<I>>::type> deserializer =
                CreateSomeipDeserializer<I>();
            typename std::decay<ArgType<I>>::type tmp = deserializer.GetValue();
            return tmp;
        } else {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->warn() <<  "Wrong serialize type, SHM deserialize type used!";
            vrtf::serialize::dds::Deserializer<typename std::decay<ArgType<I>>::type> deserializer(
                posIndex_[I], argsSize_[I], serializeConfig_);
            typename std::decay<ArgType<I>>::type tmp = deserializer.GetValue();
            return tmp;
        }
    }

    /**
     * @brief Initializes posIndex_ with the starting position of each argument in the given payload.
     *        Initializes argsSize_ with the size of each argument.
     * @details During this process the deserialization validation will be checked.
     * @param[in] Head the first parameter in recursive template.
     * @param[in] Tail the other parameters in recursive template.
     * @return true if the payload can be deserialized successfully, or false.
     */
    template<typename Head, typename... Tail>
    bool IndexMethodParameter(Head head, Tail... args)
    {
        std::size_t size = GetArgSize(head);
        if (size > remainingSize_) {
            return false;
        } else {
            posIndex_.push_back(currentPos_);
            argsSize_.push_back(size);
            if (!currentNodeConfig_.isChildNodeEnableTlv) {
                currentPos_ += size;
                remainingSize_ -= size;
            }
            return IndexMethodParameter(args...);
        }
    }

    template<typename Head>
    bool IndexMethodParameter(Head head)
    {
        std::size_t size = GetArgSize(head);
        if (size > remainingSize_) {
            return false;
        } else {
            posIndex_.push_back(currentPos_);
            argsSize_.push_back(size);
            return true;
        }
    }

    bool IndexMethodParameter() const
    {
        return true;
    }
    bool DoInit()
    {
        bool isValid = true;
        if (currentNodeConfig_.childNodeList != nullptr && !currentNodeConfig_.childNodeList->empty()) {
            childNodeView_ = currentNodeConfig_.childNodeList->begin();
            if (currentNodeConfig_.isChildNodeEnableTlv) {
                tlv::serialize::RecordDataIdResult res {tlv::serialize::TlvSerializeHelper::RecordDataId(
                    currentNodeConfig_.serializationConfig.byteOrder,
                    currentNodeConfig_.tlvLengthFieldSize, data_, remainingSize_)};
                dataIdMap_ = std::move(res.dataIdMap);
                if (res.size > remainingSize_) {
                    isValid = false;
                    const size_t logLimit = 500;
                    /* AXIVION disable style AutosarC++19_03-A5.1.1: Records the log */
                    /* AXIVION disable style AutosarC++19_03-A5.0.1: Records the log */
                    logInstance_->error("MethodSkeleton_ReplySerialize",
                    {logLimit, ara::godel::common::log::LogLimitConfig::LimitType::TIMELIMIT}) <<
                        "[ParamsDeserializer][Method tlv deserialize failed, data stream is over buffer size]";
                    /* AXIVION enable style AutosarC++19_03-A5.0.1 */
                    /* AXIVION enable style AutosarC++19_03-A5.1.1 */
                } else {
                    isValid = true;
                }
            }
        }
        return isValid;
    }
private:
    /**
     * @brief Get the size of Ith parameter.
     * @param[in] arg the Ith parameter.
     * @return the size of Ith parameter.
     */
    template<typename T>
    std::size_t GetArgSize(const T& arg)
    {
        static_cast<void>(arg);
        if (serializeType_ == vrtf::serialize::SerializeType::SHM) {
            vrtf::serialize::dds::Deserializer<typename std::decay<T>::type> deserializer(
                currentPos_, remainingSize_, serializeConfig_);
            return deserializer.GetSize();
        } else if (serializeType_ == vrtf::serialize::SerializeType::SOMEIP) {
            return DoGetSomeipSize<T>();
        } else {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->warn() <<  "Wrong serialize type, SHM deserialize type used!";
            vrtf::serialize::dds::Deserializer<typename std::decay<T>::type> deserializer(
                currentPos_, remainingSize_, serializeConfig_);
            return deserializer.GetSize();
        }
    }
    template<typename T>
    std::size_t DoGetSomeipSize()
    {
        if (currentNodeConfig_.childNodeList != nullptr && (!(currentNodeConfig_.childNodeList->empty()))) {
            std::size_t size {vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE};
            if (childNodeView_ == currentNodeConfig_.childNodeList->cend()) {
                return size;
            }
            if (currentNodeConfig_.isChildNodeEnableTlv) {
                size = vrtf::serialize::someip::DeserializeSizeCounter::GetParamSizeOfTlv<T>(dataIdMap_,
                    *childNodeView_, data_);
                if (size != vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE) {
                    currentPos_ =  data_ + dataIdMap_[(*childNodeView_)->dataId].totalSize;
                }
            } else {
                vrtf::serialize::someip::Deserializer<typename std::decay<T>::type> deserializer(
                    currentPos_, remainingSize_, *childNodeView_);
                size = deserializer.GetSize();
                if (size != vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE) {
                    uint8_t paddingSize = deserializer.GetAlignmentPaddingSize();
                    size = vrtf::serialize::someip::SerializateHelper::GetDeserializeRealSize(remainingSize_,
                        size, paddingSize, (*childNodeView_)->isLastSerializeNode);
                }
            }
            childNodeViewList_.push_back(childNodeView_);
            ++childNodeView_;
            return size;
        } else {
            vrtf::serialize::someip::Deserializer<typename std::decay<T>::type> deserializer(
                currentPos_, remainingSize_, serializeConfig_);
            return deserializer.GetSize();
        }
    }
    template<std::size_t I>
    vrtf::serialize::someip::Deserializer<typename std::decay<ArgType<I>>::type> CreateSomeipDeserializer()
    {
        if (currentNodeConfig_.childNodeList != nullptr) {
            return vrtf::serialize::someip::Deserializer<typename std::decay<ArgType<I>>::type>(
                posIndex_[I], argsSize_[I], *(childNodeViewList_[I]));
        } else {
            return vrtf::serialize::someip::Deserializer<typename std::decay<ArgType<I>>::type>(
                posIndex_[I], argsSize_[I], serializeConfig_);
        }
    }
    const vrtf::vcc::api::types::MethodMsg& msg_;
    const std::uint8_t* currentPos_;
    const std::uint8_t* data_;
    std::size_t remainingSize_;
    vrtf::serialize::SerializeType serializeType_;
    vrtf::serialize::SerializeConfig serializeConfig_;
    vrtf::serialize::SerializationNode currentNodeConfig_;
    vrtf::serialize::SerializationList::iterator childNodeView_;
    std::vector<const std::uint8_t*> posIndex_;
    std::vector<std::size_t> argsSize_;
    std::vector<vrtf::serialize::SerializationList::iterator> childNodeViewList_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    std::unordered_map<std::uint16_t, tlv::serialize::DataIdParams> dataIdMap_ {};
};

class ParamsSerializer {
public:
    ParamsSerializer(uint8_t* payloadData, vrtf::serialize::SerializeType type,
        vrtf::serialize::SerializationNode const &requestserializationNode,
        const vrtf::serialize::SerializeConfig &config = vrtf::serialize::SerializeConfig ())
        : currentPos_(payloadData), serializeType_(type), serializeConfig_(config),
          currentNodeConfig_(requestserializationNode)
    {
        using namespace ara::godel::common;
        logInstance_ = log::Log::GetLog("CM");
    }
    ~ParamsSerializer() = default;
    template <typename... Args>
    void Serialize(Args &&... args)
    {
        if (currentNodeConfig_.childNodeList != nullptr && (!currentNodeConfig_.childNodeList->empty())) {
            childNodeView_ = currentNodeConfig_.childNodeList->cbegin();
        }
        DoSerialize(currentPos_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    const std::size_t& GetSize(Args &&... args)
    {
        if (currentNodeConfig_.childNodeList != nullptr && (!currentNodeConfig_.childNodeList->empty())) {
            childNodeView_ = currentNodeConfig_.childNodeList->cbegin();
        }
        DoGetSize(std::forward<Args>(args)...);
        return size_;
    }

    void SetBuffer(uint8_t* payloadData)
    {
        currentPos_ = payloadData;
    }

private:
    uint8_t* currentPos_;  // Data vector the class will serialize into.

    vrtf::serialize::SerializeType serializeType_;
    vrtf::serialize::SerializeConfig serializeConfig_;
    vrtf::serialize::SerializationNode currentNodeConfig_;
    vrtf::serialize::SerializationList::const_iterator childNodeView_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;

    std::size_t size_ = 0;
    std::size_t pos_ {0};
    void DoSerialize(const uint8_t* payloadData) const
    {
        (void)payloadData;
    }
    void DoGetSize() const
    {
    }

    template <typename HEAD, typename... TAIL>
    void DoSerialize(uint8_t *payloadData, HEAD && head, TAIL &&... tail)
    {
        static_cast<void>(payloadData);
        if (serializeType_ == vrtf::serialize::SerializeType::SHM) {
            vrtf::serialize::dds::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), serializeConfig_);
            std::size_t size = serializer.GetSize();
            serializer.Serialize(currentPos_);
            currentPos_ += size;
            DoSerialize(currentPos_, std::forward<TAIL>(tail)...);
        } else if (serializeType_ == vrtf::serialize::SerializeType::SOMEIP) {
            DoSomeipSerialize(std::forward<HEAD>(head));
            DoSerialize(currentPos_, std::forward<TAIL>(tail)...);
        } else {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error() <<  "Wrong serialize type, serialize error!";
        }
    }
    template <typename HEAD, typename... TAIL>
    void DoGetSize(HEAD && head, TAIL &&... tail)
    {
        if (serializeType_ == vrtf::serialize::SerializeType::SHM) {
            vrtf::serialize::dds::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), serializeConfig_);
            size_ += serializer.GetSize();
            DoGetSize(std::forward<TAIL>(tail)...);
        } else if (serializeType_ == vrtf::serialize::SerializeType::SOMEIP) {
            DoGetSomeipSize(std::forward<HEAD>(head));
            DoGetSize(std::forward<TAIL>(tail)...);
        } else {
            /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
            logInstance_->error() <<  "Wrong serialize type, serialize error!";
        }
    }

    template <typename HEAD>
    void DoGetSomeipSize(HEAD && head)
    {
        if (size_ == vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE) {
            return;
        }
        if (currentNodeConfig_.childNodeList != nullptr && (!(currentNodeConfig_.childNodeList->empty()))) {
            if (childNodeView_ != currentNodeConfig_.childNodeList->cend()) {
                vrtf::serialize::someip::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), *childNodeView_,
                    size_, (*childNodeView_)->isLastSerializeNode);
                if (currentNodeConfig_.isChildNodeEnableTlv) {
                    size_ += vrtf::serialize::someip::TWO_BYTES_LENGTH; // Tag length filed
                }
                std::size_t size = serializer.GetSize();
                if (size == vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE) {
                    size_ = vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE;
                    return;
                }
                size_ += size;
                ++childNodeView_;
            } else {
                /* AXIVION Next Line AutosarC++19_03-A5.1.1, AutosarC++19_03-A5.0.1 : Records the log */
                logInstance_->error() <<  "Wrong method serialize config, serialize error!";
                size_ = vrtf::serialize::someip::MAX_SOMEIP_SERIALIZE_SIZE;
            }
        } else {
            vrtf::serialize::someip::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), serializeConfig_);
            size_ += serializer.GetSize();
        }
    }

    template <typename HEAD>
    void DoSomeipSerialize(HEAD && head)
    {
        if (currentNodeConfig_.childNodeList != nullptr && (!(currentNodeConfig_.childNodeList->empty()))) {
            if (childNodeView_ != currentNodeConfig_.childNodeList->cend()) {
                std::size_t tagLength {0};
                if (currentNodeConfig_.isChildNodeEnableTlv) {
                    tagLength = vrtf::serialize::someip::TWO_LENGTH_FIELD;
                }
                vrtf::serialize::someip::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), *childNodeView_,
                    pos_ + tagLength, (*childNodeView_)->isLastSerializeNode);
                std::size_t size = serializer.GetSize();
                if (currentNodeConfig_.isChildNodeEnableTlv) {
                    tlv::serialize::TlvSerializeHelper::CopyTagData<typename std::decay<HEAD>::type>(
                        currentPos_ + pos_, size_ - pos_, *childNodeView_, serializer.GetLengthFieldSize());
                    pos_ += tagLength;
                }
                serializer.Serialize(currentPos_);
                pos_ += size;
                ++childNodeView_;
            }
        } else {
            vrtf::serialize::someip::Serializer<typename std::decay<HEAD>::type> serializer(std::forward<HEAD>(head), serializeConfig_);
            std::size_t size = serializer.GetSize();
            serializer.Serialize(currentPos_);
            currentPos_ += size;
        }
    }
};
}
}
}

#endif
