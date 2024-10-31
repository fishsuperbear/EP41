/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: The declaration of E2EXf_PXmInfo.h
 * Create: 2021-06-10
 */

#ifndef VRTF_COM_E2EXF_PXMINFO_H
#define VRTF_COM_E2EXF_PXMINFO_H

#include <cstdint>

namespace vrtf {
namespace com {
namespace e2exf {
/**
    * @brief The message result will be protected or checked
    *
    */
enum class E2EXfMessageResult : std::uint8_t {
    OK = 0U,
    ERROR
};

/**
 * @brief The message type will be protected or checked
 *
 */
enum class E2EXfMessageType : std::uint8_t {
    REQUEST = 0U,
    RESPONSE
};

/**
 * @brief the information will be used in protect and check if using Profile Xm
 *
 */
class E2EXf_PXmInfo final {
public:
    /**
     * @brief Construct a default e2exf pxminfo object
     *
     */
    E2EXf_PXmInfo() noexcept : E2EXf_PXmInfo(E2EXfMessageResult::ERROR, E2EXfMessageType::REQUEST, UINT32_MAX, false) {};

    /**
     * @brief Construct a new e2exf pxminfo object
     *
     * @param[in] Result  The message result of the handling payload
     * @param[in] Type    The message typeof the handling payload
     */
    E2EXf_PXmInfo(const E2EXfMessageResult Result, const E2EXfMessageType Type) noexcept
        : E2EXf_PXmInfo(Result, Type, UINT32_MAX, false) {}

    /**
     * @brief Construct a new e2exf pxminfo object
     *
     * @param[in] Result        The message result of the handling payload
     * @param[in] Type          The message type of the handling payload
     * @param[in] SourceId      The sourceId will be used in the handling payload
     */
    E2EXf_PXmInfo(const E2EXfMessageResult Result, const E2EXfMessageType Type, const std::uint32_t SourceId) noexcept
        : E2EXf_PXmInfo(Result, Type, SourceId, true) {}

    /**
    * @brief Destroy the e2exf pxminfo object
    *
    */
    ~E2EXf_PXmInfo() = default;

    /**
     * @brief Construct a new e2exf pxminfo object
     *
     * @param[in] Other     other instance will be copied
     */
    E2EXf_PXmInfo(const E2EXf_PXmInfo& Other) = default;

    /**
     * @brief copy assignment constructor
     *
     * @param[in] Other     other instance will be copied
     * @return E2EXf_PXmInfo&
     */
    E2EXf_PXmInfo& operator=(const E2EXf_PXmInfo& Other) & = default;

    /**
    * @brief Set the Message Type object
    *
    * @param[in] Type    The message typeof the handling payload
    */
    void SetMessageType(const E2EXfMessageType Type) noexcept { Type_ = Type; }

    /**
    * @brief Set the Message Result object
    *
    * @param[in] Result  The message result of the handling payload
    */
    void SetMessageResult(const E2EXfMessageResult Result) noexcept { Result_ = Result; }

    /**
     * @brief Set the Source Id object
     *
     * @param[in] SourceId      The sourceId will be used in the handling payload
     */
    void SetSourceId(const std::uint32_t SourceId) noexcept
    {
        SourceId_ = SourceId;
        IsSettingSourceId_ = true;
    }

    /**
     * @brief Get the Message Type object
     *
     */
    E2EXfMessageType GetMessageType() const noexcept { return Type_; }

    /**
     * @brief Get the Message Result object
     *
     */
    E2EXfMessageResult GetMessageResult() const noexcept { return Result_; }

    /**
     * @brief Get the Source Id object
     *
     */
    std::uint32_t GetSourceId() const noexcept { return SourceId_; }

    /**
     * @brief Get wheather the sourceId is setted
     *
     */
    bool IsSettingSourceId() const noexcept { return IsSettingSourceId_; }
private:
    E2EXfMessageResult Result_;
    E2EXfMessageType Type_;
    std::uint32_t SourceId_;
    bool IsSettingSourceId_;

    E2EXf_PXmInfo(const E2EXfMessageResult Result, const E2EXfMessageType Type,
                  const std::uint32_t SourceId, const bool isSettingSourceId) noexcept
        : Result_(Result), Type_(Type), SourceId_(SourceId), IsSettingSourceId_(isSettingSourceId)
    {}
};
} /* End namespace e2exf */
} /* End namespace com */
} /* End namespace vrtf */
#endif

