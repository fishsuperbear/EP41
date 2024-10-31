/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: The declaration of E2EXf_Handler.h
 * Create: 2021-06-10
 */

#ifndef VRTF_COM_E2EXF_HANDLER_H
#define VRTF_COM_E2EXF_HANDLER_H

#include <memory>
#include <mutex>
#include <unordered_set>
#include <string>
#include <unordered_map>
#include "vrtf/com/e2e/E2EXf/ResultImpl.h"
#include "vrtf/com/e2e/E2EXf/E2EXf_CMConfig.h"
#include "vrtf/com/e2e/E2EXf/E2EXf_PXmInfo.h"

namespace vrtf {
namespace com {
namespace e2e {
namespace impl {
class E2EXf_Impl;
class E2EXf_ConfigTypeImpl;
} /* End namspace Impl */
} /* End namespace e2e */
namespace e2exf {
using SMState = vrtf::com::e2e::impl::SMStateImpl;

enum class ProfileCheckStatus : std::uint8_t {
    kOk = 0U,
    kRepeated,
    kWrongSequence,
    kError,
    kNotAvailable,
    kNoNewData,
    kCheckDisabled
};

class Result final {
public:
    /**
     * @brief Construct a new Result object
     *
     */
    Result() noexcept : Result(ProfileCheckStatus::kNotAvailable, SMState::kStateMDisabled) {}

    /**
     * @brief Construct a new Result object
     *
     * @param[in] CheckStatus   represents the results of the check of a single sample
     * @param[in] State         represents in what state is the E2E supervision after the most recent check of
     *                          the sample(s) of a received sample of the event.
     */
    Result(ProfileCheckStatus CheckStatus, SMState State) noexcept : CheckStatus_(CheckStatus), SMState_(State) {}

    /**
     * @brief Destroy the Result object
     *
     */
    ~Result() = default;

    /**
     * @brief Construct a new Result object
     *
     * @param Other[in]     The other instance will be copied
     */
    Result(const Result& Other) noexcept : CheckStatus_(Other.CheckStatus_), SMState_(Other.SMState_) {}

    /**
     * @brief The copy assignment constructor
     *
     * @param[in] Other          The other instance will be copied
     * @return Result&          The new instance which contains the same value with input instance
     */
    Result& operator=(const Result& Other) & noexcept;

    /**
     * @brief Get the Profile Check Status
     *
     * @return ProfileCheckStatus  represents the results of the check of a single sample
     */
    ProfileCheckStatus GetProfileCheckStatus() const noexcept { return CheckStatus_;}

    /**
     * @brief
     *
     * @return SMState  represents in what state is the E2E supervision after the most recent check of the sample(s) of
     *                  a received sample of the event.
     */
    SMState GetSMState() const noexcept { return SMState_; }
private:
    ProfileCheckStatus CheckStatus_;
    SMState SMState_;
};
/* AXIVION Next Line AutosarC++19_03-A0.1.6 : public interface */
class E2EXf_Handler final {
public:
    /**
     * @brief Construct a new e2exf handler object
     *
     */
    E2EXf_Handler() = default;

    /**
     * @brief Destroy the e2exf handler object
     *
     */
    ~E2EXf_Handler();

    /**
     * @brief Initialize the handler
     *
     * @param[in] CMConfig      The Config will be used
     *
     * @return bool
     *      @retval true    Initialize success
     *      @retval false   Initialize failed
     */
    bool Init(E2EXf_CMConfig const &CMConfig);

    /**
     * @brief Common E2E protect
     *
     * @param[inout] Buffer    The Buffer points to the data will be protect
     *                          and the buffer contains unused area for Header
     * @param[in] BufferLength  The length of used buffer, that is not including Header
     *
     * @return bool     operator result
     *      @retval true    E2E Protect success
     *      @retval false   E2E Protect failed
     */
    bool Protect(std::uint8_t* Buffer, const std::uint32_t BufferLength);

    /**
     * @brief E2E Protect will be used in using specific counter or incorrect Id
     *
     * @param[inout] Buffer    The Buffer points to the data will be protect
     *                          and the buffer contains unused area for Header
     * @param[in] BufferLength          The length of used buffer, that is not including Header
     * @param[in] Counter               The Counter will be used in Protect
     * @param[in] IsUsingIncorrectId    If using incorrect Id
     *
     * @return bool     operator result
     *      @retval true    E2E Protect success
     *      @retval false   E2E Protect failed
     */
    bool Protect(std::uint8_t* Buffer, const std::uint32_t BufferLength,
                 const std::uint32_t Counter, bool IsUsingIncorrectId = false);
    /**
    * @brief  Common E2E protect when using Profile Xm
    *
    * @param[inout] Buffer      The Buffer points to the data will be protect
     *                          and the buffer contains unused area for Header
    * @param[in] BufferLength   The length of used buffer, that is not including Header
    * @param ProtectInfo        The ProfileXm info will be used in protect
    *
    * @return bool     operator result
     *      @retval true    E2E Protect success
     *      @retval false   E2E Protect failed
    */
    bool ProtectPXm(std::uint8_t* Buffer, const std::uint32_t BufferLength, const E2EXf_PXmInfo& ProtectInfo);

    /**
     * @brief E2E Protect will be used in using specific counter or incorrect Id if using Profile Xm
     *
     * @param[inout] Buffer             The Buffer points to the data will be protect
     *                                  and the buffer contains unused area for Header
     * @param[in] BufferLength          The length of used buffer, that is not including Header
     * @param[in] Counter               The Counter will be used in Protect
     * @param ProtectInfo               The ProfileXm info will be used in protect
     * @param[in] IsUsingIncorrectId    If using incorrect Id
     *
     * @return bool     operator result
     *      @retval true    E2E Protect success
     *      @retval false   E2E Protect failed
     */
    bool ProtectPXm(std::uint8_t* Buffer, const std::uint32_t BufferLength, const std::uint32_t Counter,
        const E2EXf_PXmInfo& ProtectInfo, bool IsUsingIncorrectId = false);

    /**
     * @brief Checks the buffer to be transmitted for in-place transmission using pointer
     *
     * @param[inout] Buffer         The received data which contains E2E Header(which will be removed by E2E)
     * @param[in] BufferLength      The length of user data
     *
     * @return Result   return corresponding ProfileCheckStatus & SMState
     */
    Result Check(std::uint8_t* Buffer, const std::uint32_t BufferLength);

    /**
     * @brief Checks the buffer to be transmitted for in-place transmission using pointer
     *
     * @param[inout] Buffer             The received data which contains E2E Header(which will be removed by E2E)
     * @param[in] BufferLength          The length of user data
     * @param[in] Counter               The counter will be used in check
     * @return Result    return corresponding ProfileCheckStatus & SMState
     */
    Result Check(std::uint8_t* Buffer, const std::uint32_t BufferLength, const std::uint32_t Counter);

    /**
     * @brief Checks the buffer to be transmitted for in-place transmission using pointer if using Profile Xm in client
     *
     * @param[inout] Buffer         The received data which contains E2E Header(which will be removed by E2E)
     * @param[in] BufferLength      The length of user data
     * @param[in] CheckInfo         The ProfileXm info will be used in check
     * @return Result   return corresponding ProfileCheckStatus & SMState
     */
    Result CheckPXm(std::uint8_t* Buffer, const std::uint32_t BufferLength, const E2EXf_PXmInfo& CheckInfo);

    /**
     * @brief Checks the buffer to be transmitted for in-place transmission using pointer if using Profile Xm in server
     *
     * @param[inout] Buffer         The received data which contains E2E Header(which will be removed by E2E)
     * @param[in] BufferLength      The length of user data
     * @param[in] Counter           The counter will be usedin check
     * @param[inout] CheckInfo      The ProfileXm info will be used in check, the source Id will be stored by user
     * @return Result   return corresponding ProfileCheckStatus & SMState
     */
    Result CheckPXm(std::uint8_t* Buffer, const std::uint32_t BufferLength,
                    const std::uint32_t Counter, E2EXf_PXmInfo& CheckInfo);

    /**
     * @brief Get the E2E Header size
     *
     * @return std::uint32_t           The length of E2E Header
     */
    std::uint32_t GetHeaderSize() const;

    /**
     * @brief Get the counter of the received buffer
     *
     * @param[in] Buffer            The buffer contains E2E header
     * @param[in] BufferLength      The length of buffer
     * @return std::pair<bool, std::uint32_t>   The first value represents the valid status of index,
     *                                          the second value represents the obtained counter
     */
    std::pair<bool, std::uint32_t> GetReceivedCounter(const std::uint8_t* Buffer,
                                                      const std::uint32_t BufferLength) const;

    /**
     * @brief Get the Config is used
     *
     * @return const std::shared_ptr<vrtf::com::e2exf::E2EXf_CMConfig>
     */
    std::shared_ptr<const vrtf::com::e2exf::E2EXf_CMConfig> GetCMConfig() const noexcept { return CMConfig_; }

    /**
     * @brief Get the Counter of ProtectState in Handler
     *
     * @return std::pair<bool. std::uing32_t>
     *      @retval first   If the operation is success
     *      @retval second  The current counter value in protect state
     */
    std::pair<bool, std::uint32_t> GetProtectCounter() const;

    /**
     * @brief Get the RequestCounter of ProfileXm
     *
     * @return std::pair<bool. std::uing32_t>
     *      @retval first   If the operation is success
     *      @retval second  The current request counter value of ProfileXm
     */
    std::pair<bool, std::uint32_t> GetRequestCounter() const;

    /**
     * @brief Reset the Counter by Decreasing 1 with no lock
     *
     * @return bool
     */
    bool DereaseProtectCounter() const;

private:
    std::shared_ptr<vrtf::com::e2e::impl::E2EXf_Impl> ProtectHandler_ {nullptr};
    std::shared_ptr<vrtf::com::e2e::impl::E2EXf_Impl> CheckHandler_ {nullptr};
    std::shared_ptr<vrtf::com::e2exf::E2EXf_CMConfig> CMConfig_ {nullptr};
    std::mutex ProtectMutex_;
    std::mutex CheckMutex_;

    static std::unique_ptr<e2e::impl::E2EXf_ConfigTypeImpl> CreateCommonProfile(E2EXf_CMConfig const &CMConfig);
};
} /* End namespace e2exf */
} /* End namespace com */
} /* End namespace vrtf */
#endif

