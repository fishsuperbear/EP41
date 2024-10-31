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
 * @file error_code.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_ERROR_CODE_H_
#define APD_ARA_CORE_ERROR_CODE_H_

#include <ara/core/error_domain.h>
#include <ara/core/string_view.h>

#include <cstdint>
#include <ostream>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @class ErrorCode
 *
 * @brief Class for ErrorCode
 *
 * @uptrace{SWS_CORE_00501}
 */
class ErrorCode final {
   public:
    /**
     * @brief print the detail of the ErrorCode.
     *
     * @param out [in] std::ostream object
     * @param ex [in] the ErrorCode
     * @return std::ostream object
     *
     * @uptrace{SWS_CORE_00581}
     */
    friend std::ostream &operator<<( std::ostream &out, ErrorCode const &e );

    using CodeType        = ErrorDomain::CodeType;
    using SupportDataType = ErrorDomain::SupportDataType;

    /**
     * @brief Construct a new ErrorCode instance with parameters.
     *
     * @tparam EnumT an enum type that contains error code values
     * @param e  [in] a domain-specific error code value
     * @param data  [in] optional vendor-specific supplementary error context data
     *
     * @uptrace{SWS_CORE_00512}
     */
    template <typename EnumT>
    constexpr ErrorCode( EnumT e, SupportDataType data = ErrorDomain::SupportDataType() ) noexcept
        // Call MakeError() unqualified, so the correct overload is found via ADL.
        : ErrorCode( MakeErrorCode( e, data ) ) {}

    /**
     * @brief Construct a new ErrorCode instance with parameters.
     *
     * @param value  [in] a domain-specific error code value
     * @param domain  [in] the ErrorDomain associated with value
     * @param data  [in] optional vendor-specific supplementary error context data
     *
     * @uptrace{SWS_CORE_00513}
     */
    constexpr ErrorCode(
        ErrorDomain::CodeType value, ErrorDomain const &domain,
        ErrorDomain::SupportDataType data = ErrorDomain::SupportDataType() ) noexcept
        : mValue( value ), mSupportData( data ), mDomain( &domain ) {}

    /**
     * @brief Return the raw error code value.
     *
     * @return the raw error code value
     *
     * @uptrace{SWS_CORE_00514}
     */
    constexpr CodeType Value() const noexcept { return mValue; }

    /**
     * @brief Return the domain with which this ErrorCode is associated.
     *
     * @return the ErrorDomain
     *
     * @uptrace{SWS_CORE_00515}
     */
    constexpr ErrorDomain const &Domain() const noexcept { return *mDomain; }

    /**
     * @brief Return the supplementary error context data.
     *
     * @return the supplementary error context data
     *
     * @uptrace{SWS_CORE_00516}
     */
    constexpr SupportDataType SupportData() const noexcept { return mSupportData; }

    /**
     * @brief Return a textual representation of this ErrorCode.
     *
     * @return the error message text
     *
     * @uptrace{SWS_CORE_00518}
     */
    StringView Message() const noexcept;

    /**
     * @brief Throw this error as exception.
     *
     * @uptrace{SWS_CORE_00519}
     */
    void ThrowAsException() const;

   private:
    CodeType           mValue;
    SupportDataType    mSupportData;
    ErrorDomain const *mDomain;  // non-owning pointer to the associated ErrorDomain
};

/**
 * @brief Global operator== for ErrorCode.
 *
 * @param lhs [in] the left hand side of the comparison
 * @param rhs [in] the right hand side of the comparison
 * @return true if the two instances compare equal, false otherwise
 *
 * @uptrace{SWS_CORE_00571}
 */
constexpr bool operator==( ErrorCode const &lhs, ErrorCode const &rhs ) noexcept {
    return lhs.Domain() == rhs.Domain() && lhs.Value() == rhs.Value();
}

/**
 * @brief Global operator!= for ErrorCode.
 *
 * @param lhs [in] the left hand side of the comparison
 * @param rhs [in] he right hand side of the comparison
 * @return true if the two instances compare not equal, false otherwise
 *
 * @uptrace{SWS_CORE_00572}
 */
constexpr bool operator!=( ErrorCode const &lhs, ErrorCode const &rhs ) noexcept {
    return lhs.Domain() != rhs.Domain() || lhs.Value() != rhs.Value();
}

/**
 * @brief Throws the given errorCode as Exception
 * @param errorCode error code to be thrown
 * @remark if ARA_NO_EXCEPTIONS is defined, this function call will terminate.
 */
template <typename ExceptionType>
void ThrowOrTerminate( ErrorCode errorCode ) {
#ifndef ARA_NO_EXCEPTIONS
    throw ExceptionType( std::move( errorCode ) );
#else
    (void) errorCode;
    std::terminate();
#endif
}
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_ERROR_CODE_H_
/* EOF */
