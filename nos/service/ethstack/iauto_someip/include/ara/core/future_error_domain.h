/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file future_error_domain.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_FUTURE_ERROR_DOMAIN_H
#define APD_ARA_CORE_FUTURE_ERROR_DOMAIN_H

#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>
#include <ara/core/exception.h>

#include <cstdint>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @brief Specifies the types of internal errors that can occur upon calling
 * Future::get or Future::GetResult.
 *
 * These definitions are equivalent to the ones from std::future_errc.
 *
 * @uptrace{SWS_CORE_00400}
 */
enum class future_errc : int32_t {
    broken_promise            = 101,  ///< the asynchronous task abandoned its shared state
    future_already_retrieved  = 102,  ///< the contents of the shared state were already accessed
    promise_already_satisfied = 103,  ///< attempt to store a value into the shared state twice
    no_state = 104,  ///< attempt to access Promise or Future without an associated state
    unknow_error
};

/**
 * @brief Exception type thrown by Future and Promise classes.
 *
 * @uptrace{SWS_CORE_00411}
 */
class FutureException : public Exception {
   public:
    /**
     * @brief Construct a new FutureException from an ErrorCode.
     * @param err  the ErrorCode
     *
     * @uptrace{SWS_CORE_00412}
     */
    explicit FutureException( ErrorCode err ) noexcept;
};

/**
 * @brief Error domain for errors originating from classes Future and Promise.
 *
 * @uptrace{SWS_CORE_00421}
 */
class FutureErrorDomain final : public ErrorDomain {
   public:
    /**
     * @brief Alias for the error code value enumeration
     *
     * @uptrace{SWS_CORE_00431}
     */
    using Errc = future_errc;

    /**
     * @brief Alias for the exception base class
     *
     * @uptrace{SWS_CORE_00432}
     */
    using Exception = FutureException;

    /**
     * @brief Default constructor
     *
     * @uptrace{SWS_CORE_00441}
     */
    constexpr FutureErrorDomain() noexcept : ErrorDomain( kId ) {}

    /**
     * @brief Return the "shortname" ApApplicationErrorDomain.SN of this error domain.
     *
     * @return "Future"
     *
     * @uptrace{SWS_CORE_00442}
     */
    char const *Name() const noexcept override;

    /**
     * @brief Translate an error code value into a text message.
     *
     * @param errorCode [in] errorCode  the error code value
     * @return the text message, never nullptr
     *
     * @uptrace{SWS_CORE_00443}
     */
    char const *Message( ErrorDomain::CodeType errorCode ) const noexcept override;

    /**
     * @brief Throw the exception type corresponding to the given ErrorCode.
     *
     * @param errorCode the ErrorCode instance
     *
     * @uptrace{SWS_CORE_00444}
     */
    void ThrowAsException( ErrorCode const &errorCode ) const noexcept( false ) override;

   private:
    constexpr static ErrorDomain::IdType kId = 0x8000000000000013;
};

namespace internal {
/**
 * @brief the single global FutureErrorDomain instance.
 *
 */
constexpr FutureErrorDomain g_futureErrorDomain;
}  // namespace internal

/**
 * @brief Obtain the reference to the single global FutureErrorDomain instance.
 * @returns reference to the FutureErrorDomain instance
 *
 * @uptrace{SWS_CORE_00480}
 */
constexpr ErrorDomain const &GetFutureDomain() noexcept { return internal::g_futureErrorDomain; }

/**
 * @brief Create a new ErrorCode for FutureErrorDomain with the given support
 * data type.
 *
 * @param code  an enumeration value from future_errc
 * @param data  a vendor-defined supplementary value
 * @returns the new ErrorCode instance
 *
 * @uptrace{SWS_CORE_00490}
 */
constexpr ErrorCode MakeErrorCode( future_errc code, ErrorDomain::SupportDataType data ) noexcept {
    return ErrorCode( static_cast<ErrorDomain::CodeType>( code ), GetFutureDomain(), data );
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_FUTURE_ERROR_DOMAIN_H
/* EOF */
