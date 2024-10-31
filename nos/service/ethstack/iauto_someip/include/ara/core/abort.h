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
 * @file abort.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_ABORT_H_
#define APD_ARA_CORE_ABORT_H_

namespace ara {
namespace core {
inline namespace _19_11 {
// @uptrace{SWS_CORE_00050}
using AbortHandler = void ( * )();

/**
 * @brief Set a custom global Abort handler function and return the previously installed one.
 *
 * @param handler [in] a custom Abort handler (or nullptr)
 * @return the previously installed Abort handler (or nullptr if none was installed)
 *
 * @uptrace{SWS_CORE_00051}
 */
AbortHandler SetAbortHandler( AbortHandler handler ) noexcept;

// template <class Func> AbortHandler call(Func f) { return f(); }

/**
 * @brief Terminate the current process abnormally.
 *
 * @param text [in] a custom text to include in the log message being output
 *
 * @uptrace{SWS_CORE_00052}
 */
void Abort( char const *text ) noexcept;
}  // namespace _19_11
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_ABORT_H_
/* EOF */
