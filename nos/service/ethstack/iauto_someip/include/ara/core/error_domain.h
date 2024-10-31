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
 * @file error_domain.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_ERROR_DOMAIN_H_
#define APD_ARA_CORE_ERROR_DOMAIN_H_

#include <cstdint>

namespace ara {
namespace core {
inline namespace _19_11 {
// forward declaration
class ErrorCode;

/**
 * @class ErrorDomain
 *
 * @brief Encapsulation of an error domain.
 *
 * @uptrace{SWS_CORE_00110}
 */
class ErrorDomain {
   public:
    /// @uptrace{SWS_CORE_00121}
    using IdType = std::uint64_t;
    /// @uptrace{SWS_CORE_00122}
    using CodeType = std::int32_t;
    /// @uptrace{SWS_CORE_00123}
    using SupportDataType = std::int32_t;

    /// @uptrace{SWS_CORE_00131}
    ErrorDomain( ErrorDomain const & ) = delete;
    /// @uptrace{SWS_CORE_00132}
    ErrorDomain( ErrorDomain &&rh ) = delete;

    /// @uptrace{SWS_CORE_00133}
    ErrorDomain &operator=( ErrorDomain const & ) = delete;
    /// @uptrace{SWS_CORE_00134}
    ErrorDomain &operator=( ErrorDomain && ) = delete;

    /**
     * @brief Return the name of this error domain.
     *
     * @return the name as a null-terminated string, never nullptr
     *
     * @uptrace{SWS_CORE_00152}
     */
    virtual char const *Name() const noexcept = 0;

    /**
     * @brief Return a textual representation of the given error code.
     *
     * @param errorCode [in] the domain-specific error code
     * @return the text as a null-terminated string, never nullptr
     *
     * @uptrace{SWS_CORE_00153}
     */
    virtual char const *Message( CodeType errorCode ) const noexcept = 0;

    /**
     * @brief Throws the given errorCode as Exception
     * @param errorCode error code to be thrown
     * @remark if ARA_NO_EXCEPTIONS is defined, this function call will terminate.
     *
     * @uptrace{SWS_CORE_00154}
     */
    virtual void ThrowAsException( ErrorCode const &errorCode ) const noexcept( false ) = 0;

    /**
     * @brief Return the unique domain identifier.
     *
     * @return the identifier
     *
     * @uptrace{SWS_CORE_00151}
     */
    constexpr IdType Id() const noexcept { return mId; }

    /**
     * @brief Compare for equality with another ErrorDomain instance.
     * @param other [in] the other instance
     * @return true if other is equal to *this, false otherwise
     *
     * @uptrace{SWS_CORE_00137}
     */
    constexpr bool operator==( ErrorDomain const &other ) const noexcept {
        return mId == other.mId;
    }

    /**
     * @brief Compare for non-equality with another ErrorDomain instance.
     * @param other [in] the other instance
     * @return true if other is not equal to *this, false otherwise
     *
     * @uptrace{SWS_CORE_00138}
     */
    constexpr bool operator!=( ErrorDomain const &other ) const noexcept {
        return mId != other.mId;
    }

   protected:
    /**
     * @brief Construct a new instance with the given identifier.
     * @param id [in] the unique identifier
     *
     * @uptrace{SWS_CORE_00135}
     */
    explicit constexpr ErrorDomain( IdType id ) noexcept : mId( id ) {}

    /**
     * @brief Destructor.
     *
     * @uptrace{SWS_CORE_00136}
     */
    ~ErrorDomain() = default;

   private:
    IdType const mId;
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_ERROR_DOMAIN_H_
/* EOF */
