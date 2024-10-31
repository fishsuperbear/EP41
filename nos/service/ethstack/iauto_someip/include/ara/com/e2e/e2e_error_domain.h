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
 * @file e2e_error_domain.h
 * @brief
 * @date 2020-05-09
 *
 */

#ifndef INCLUDE_COM_E2E_E2E_ERROR_DOMAIN_H_
#define INCLUDE_COM_E2E_E2E_ERROR_DOMAIN_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>
#include <ara/core/exception.h>

/** @uptrace{SWS_CM_10474} */
namespace ara {
namespace com {
namespace e2e {
inline namespace _19_11 {

// @uptrace{SWS_CM_10474}
enum class E2EError : ara::core::ErrorDomain::CodeType {
    repeated             = 1,  ///< Data has a repeated counter.
    wrong_sequence_error = 2,  ///< The checks of the sample in this cycle were successful, with the exception of
                               ///< counter jump, which changed more than the allowed delta.
    error                = 3,  ///< Error not related to counters occurred (e.g. wrong crc, wrong length, wrong
                               ///< Data ID).
    not_available        = 4,  ///< No value has been received yet (e.g. during initialization). This is used as the
                               ///< initialization value for the buffer, it is not returned by any E2E profile.
    no_new_data          = 5   ///< No new data is available.
};

class E2EErrorDomainException : public ara::core::Exception {
   public:
    /**
     * @brief Construct a new E2EErrorDomainException from an ErrorCode.
     * @param err  the ErrorCode
     *
     */
    explicit E2EErrorDomainException( ara::core::ErrorCode err ) noexcept
        : ara::core::Exception( err ) {}
};

class E2EErrorDomain final : public ara::core::ErrorDomain {
    /** @uptrace{SWS_CM_11267} */
    constexpr static ara::core::ErrorDomain::IdType kId = 0x8000000000001266;

   public:
    // @brief Alias for the error code value enumeration.
    using Errc = E2EError;
    // @brief Alias for the exception base class.
    using Exception = E2EErrorDomainException;

    /**
     * @brief Default constructor
     *
     * @uptrace{SWS_CORE_10474}
     */
    constexpr E2EErrorDomain() noexcept : ara::core::ErrorDomain( kId ) {}

    /**
     * @brief Return the "shortname" of this error domain.
     *
     * @return char const* the "shortname" of this error domain.
     */
    char const* Name() const noexcept override { return "E2E"; }

    /**
     * @brief Translate an error code value into a text message.
     *
     * @param errorCode the error code value.
     * @return char const* the text message, never nullptr.
     */
    char const* Message( ara::core::ErrorDomain::CodeType errorCode ) const noexcept override {
        Errc const code = static_cast<Errc>( errorCode );
        switch ( code ) {
            case Errc::repeated:
                return "Data has a repeated counter.";
            case Errc::wrong_sequence_error:
                return "The checks of the Data in this cycle were successful, with the exception "
                       "of counter jump, which changed more than the allowed delta. ";
            case Errc::error:
                return "Error not related to counters occurred (e.g. wrong crc, wrong length, "
                       "wrong Data ID).";
            case Errc::not_available:
                return "No value has been received yet (e.g. during initialization). This is used "
                       "as the initialization value for the buffer, it is not returned by any E2E "
                       "profile.";
            case Errc::no_new_data:
                return "No new data is available.";
            default:
                return "Unknown error";
        }
    }

    /**
     * @brief Throw the exception type corresponding to the given ErrorCode.
     *
     * @param errorCode the ErrorCode instance.
     */
    void ThrowAsException( ara::core::ErrorCode const& errorCode ) const
        noexcept( false ) override {
        ara::core::ThrowOrTerminate<Exception>( errorCode );
    }
};

namespace internal {
    constexpr E2EErrorDomain g_E2EErrorDomain;
}

/**
 * @brief Obtain the reference to the single global E2EErrorDomain instance.
 * @returns reference to the E2EErrorDomain instance
 *
 */
inline constexpr ara::core::ErrorDomain const& GetE2EErrorDomain() noexcept {
    return internal::g_E2EErrorDomain;
}

/**
 * @brief Create a new ErrorCode for E2EErrorDomain with the given support data type and message.
 *
 * @param code an enumeration value from E2EError
 * @param data a vendor-defined supplementary value
 * @return constexpr ara::core::ErrorCode
 */
inline constexpr ara::core::ErrorCode MakeErrorCode(
    E2EError code, ara::core::ErrorDomain::SupportDataType data ) noexcept {
    return ara::core::ErrorCode( static_cast<ara::core::ErrorDomain::CodeType>( code ),
                                 GetE2EErrorDomain(), data );
}

}  // inline namespace _19_11
}  // namespace e2e
}  // namespace com
}  // namespace ara

#endif  // INCLUDE_COM_E2E_E2E_ERROR_DOMAIN_H_
/* EOF */
