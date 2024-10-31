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
 * @file exception.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_EXCEPTIONS_H
#define APD_ARA_CORE_EXCEPTIONS_H

#include <ara/core/error_code.h>

#include <exception>

namespace ara {
namespace core {
inline namespace _19_11 {
/**
 * @class Exception
 *
 * @brief Base type for all AUTOSAR exception types
 *
 * @uptrace{SWS_CORE_00601}
 */
class Exception : public std::exception {
   public:
    /**
     * @brief Construct a new Exception object with a specific ErrorCode.
     *
     * @param err [in] the ErrorCode
     *
     * @uptrace{SWS_CORE_00611}
     */
    explicit Exception( ErrorCode err ) noexcept;

    /**
     * @brief Return the explanatory string.
     *
     * @return a null-terminated string
     *
     * @uptrace{SWS_CORE_00612}
     */
    char const *what() const noexcept override;

    /**
     * @brief Return the embedded ErrorCode that was given to the constructor.
     *
     * @return reference to the embedded ErrorCode
     *
     * @uptrace{SWS_CORE_00613}
     */
    ErrorCode const &Error() const noexcept;

   private:
    ErrorCode const mErrorCode;
};
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_EXCEPTIONS_H
/* EOF */
