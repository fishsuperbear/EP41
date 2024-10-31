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
 * @file core_error_domain.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_CODE_ERROR_DOMAIN_H_
#define APD_ARA_CORE_CODE_ERROR_DOMAIN_H_

#include <ara/core/error_code.h>
#include <ara/core/error_domain.h>
#include <ara/core/exception.h>

#include <cstdint>

namespace ara {
namespace core {
inline namespace _19_11 {

/**
 * @enum CoreErrc
 *
 * @brief An enumeration that defines all errors of the CORE Functional Cluster.
 *
 * @uptrace{SWS_CORE_05200}
 */
enum class CoreErrc : ErrorDomain::CodeType {
    kInvalidArgument           = 22U,  /**< an invalid argument was passed to a function */
    kInvalidMetaModelShortname = 137U, /**< given string is not a valid model element shortname */
    kInvalidMetaModelPath      = 138U  /**< missing or invalid path to model element */
};

/**
 * @class CoreException
 *
 * @brief Class for CoreException
 *
 * @uptrace{SWS_CORE_05211}
 */
class CoreException : public Exception {
   public:
    /**
     * @brief Construct a new FutureException from an ErrorCode.
     * @param err  the ErrorCode
     *
     * @uptrace{SWS_CORE_05212}
     */
    explicit CoreException( ErrorCode err ) noexcept;
};

class ErrorDomain;

/**
 * @brief An error domain for errors originating from the CORE Functional Cluster.
 *
 * @uptrace{SWS_CORE_05221}
 */
class CoreErrorDomain final : public ErrorDomain {
   public:
    /**
     * @brief Alias for the error code value enumeration
     *
     * @uptrace{SWS_CORE_05231}
     */
    using Errc = CoreErrc;

    /**
     * @brief Alias for the exception base class
     *
     * @uptrace{SWS_CORE_05232}
     */
    using Exception = CoreException;

    /**
     * @brief Default constructor
     *
     * @uptrace{SWS_CORE_05241}
     */
    constexpr CoreErrorDomain() noexcept : ErrorDomain( kId ) {}

    /**
     * @brief Return the "shortname" ApApplicationErrorDomain.SN of this error domain.
     *
     * @returns "Future"
     *
     * @uptrace{SWS_CORE_05242}
     */
    char const *Name() const noexcept override;

    /**
     * @brief Translate an error code value into a text message.
     *
     * @param errorCode  the error code value
     * @returns the text message, never nullptr
     *
     * @uptrace{SWS_CORE_05243}
     */
    char const *Message( ErrorDomain::CodeType errorCode ) const noexcept override;

    /**
     * @brief Throw the exception type corresponding to the given ErrorCode.
     *
     * @param errorCode  the ErrorCode instance
     *
     * @uptrace{SWS_CORE_05244}
     */
    void ThrowAsException( ErrorCode const &errorCode ) const override;

   private:
    constexpr static ErrorDomain::IdType kId = 0x8000000000000014U;
};

namespace internal {
constexpr CoreErrorDomain g_CoreErrorDomain;
}

/**
 * @brief Obtain the reference to the single global CoreErrorDomain instance.
 * @returns reference to the CoreErrorDomain instance
 *
 * @uptrace{SWS_CORE_05280}
 */
constexpr ErrorDomain const &GetCoreErrorDomain() noexcept { return internal::g_CoreErrorDomain; }

/**
 * @brief Create a new ErrorCode for CoreErrorDomain with the given support data
 * type.
 *
 * @param code  an enumeration value from future_errc
 * @param data  a vendor-defined supplementary value
 * @returns the new ErrorCode instance
 *
 * @uptrace{SWS_CORE_05290}
 */
constexpr ErrorCode MakeErrorCode( CoreErrc code, ErrorDomain::SupportDataType data ) noexcept {
    return ErrorCode( static_cast<ErrorDomain::CodeType>( code ), GetCoreErrorDomain(), data );
}

}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_CODE_ERROR_DOMAIN_H_
/* EOF */
