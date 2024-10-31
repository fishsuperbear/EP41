/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: Publisher class
 * Create: 2020-04-22
 */
#ifndef RTF_COM_PUBLISHER_H
#define RTF_COM_PUBLISHER_H

#include "rtf/com/adapter/ros_skeleton_adapter.h"
#include "vrtf/vcc/api/raw_buffer.h"

namespace rtf {
namespace com {
template <class EventDataType>
class Publisher {
public:
    /**
     * @brief Publisher default constructor
     */
    Publisher(void) = default;

    /**
     * @brief Publisher constructor
     * @param[in] adapter The actual adapter that handles this publisher
     */
    explicit Publisher(const std::shared_ptr<adapter::RosSkeletonAdapter>& adapter) noexcept
        : adapter_(adapter)
    {
    }

    /**
     * @brief Publisher copy constructor
     * @note deleted
     * @param[in] other    Other instance
     */
    Publisher(const Publisher& other) = delete;

    /**
     * @brief Publisher move constructor
     * @param[in] other    Other instance
     */
    Publisher(Publisher && other) noexcept
        : adapter_(std::move(other.adapter_))
    {
    }

    /**
     * @brief Publisher copy assign operator
     * @note deleted
     * @param[in] other    Other instance
     */
    Publisher& operator=(const Publisher& other) = delete;

    /**
     * @brief Publisher move assign operator
     * @param[in] other    Other instance
     */
    Publisher& operator=(Publisher && other) noexcept
    {
        if (&other != this) {
            adapter_ = std::move(other.adapter_);
        }
        return *this;
    }

    /**
     * @brief Publisher default destructor
     */
    ~Publisher(void) = default;

    /**
     * @brief Returns whether of the publisher is created correctly
     * @return Is the publisher created correctly
     */
    operator bool() const noexcept
    {
        bool result{false};
        if (adapter_ != nullptr) {
            result = adapter_->IsInitialized();
        }
        return result;
    }

    /**
     * @brief Publish an event
     * @param[in] message message instance
     * @return void
     */
    void Publish(const EventDataType& message) noexcept
    {
        if (adapter_ != nullptr) {
            auto sampleInfo = CreateSampleInfo();
            static_cast<void>(adapter_->Publish<EventDataType>(message, sampleInfo));
        }
    }

    /**
     * @brief Close the connection
     * @return void
     */
    void Shutdown(void) noexcept
    {
        if (adapter_ != nullptr) {
            adapter_->Shutdown();
        }
    }

    /**
     * @brief Allocate Buffer for using RawMemory
     *
     * @param[in] size    The size of buffer will be allocated
     * @return RawMemory  The buffer was allocated
     */
    RawMemory AllocateRawMemory(std::size_t size) noexcept
    {
        if (adapter_ != nullptr) {
            return adapter_->AllocateRawMemory(size);
        }
        return RawMemory(nullptr, 0U);
    }

    /**
     * @brief  Free the allocated buffer last time
     *
     * @param[inout] buffer  the buffer will be free
     */
    void DeallocateRawMemory(RawMemory && buffer) noexcept
    {
        if (adapter_ != nullptr) {
            adapter_->DeallocateRawMemory(buffer);
        }
    }

    /**
     * @brief Publish a raw buffer
     *
     * @param[inout] buffer The buffer will be sent
     */
    void PubRawMemory(RawMemory && buffer) noexcept
    {
        if (adapter_ != nullptr) {
            static_cast<void>(adapter_->PubRawMemory(buffer));
        }
    }

    /**
     * @brief Get someip app init state, only used by rtf_tools
     * @return someip app state
     */
    rtf::com::AppState GetSomeipAppInitState() const noexcept
    {
        if (adapter_ != nullptr) {
            return adapter_->GetSomeipAppInitState();
        }
        return rtf::com::AppState::APP_DEREGISTERED;
    }

    /**
     * @brief Check all cache data is acked
     * Internal function!Prohibit to use by Application!
     *
     * @return data ack result
     */
    ReturnCode WaitForFlush(const std::uint32_t waitMs) noexcept
    {
        if (adapter_ != nullptr) {
            return adapter_->WaitForFlush(waitMs);
        }
        return rtf::com::ReturnCode::ERROR;
    }

private:
    internal::SampleInfoImpl CreateSampleInfo() const
    {
        internal::SampleInfoImpl sampleInfo;
        if (internal::PlogInfo::GetSendFlag()) {
            sampleInfo.plogInfo_ = internal::PlogInfo::CreatePlogInfo(vrtf::vcc::utils::CM_SEND);
            if (sampleInfo.plogInfo_ != nullptr) {
                sampleInfo.plogInfo_->WriteTimeStamp(
                    internal::PlogServerTimeStampNode::USER_SEND_EVENT, internal::PlogDriverType::COMMON);
            }
        }
        return sampleInfo;
    }
    std::shared_ptr<adapter::RosSkeletonAdapter> adapter_;
};
} // namespace com
} // namespace rtf
#endif // RTF_COM_PUBLISHER_H
