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
 * @file initialization.h
 * @brief
 * @date 2020-05-07
 *
 */

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef APD_ARA_CORE_INITIALIZATION_H_
#define APD_ARA_CORE_INITIALIZATION_H_

#include "ara/core/result.h"

namespace ara {
namespace core {
inline namespace _19_11 {

/**
 * @brief Initializes data structures and threads of the AUTOSAR Adaptive Runtime for Applications.
 *
 * @return A Result object that indicates whether theAUTOSAR Adaptive Runtime for Applications was
 * successfully initialized.
 * @remark this is the only way
 * for the ARA to report an error that is guaranteed to
 * be available, e.g., in case ara::log failed to correctly
 * initialize. The user is not expected to be able to
 * recover from such an error. However, the user may
 * have a project-specific way of recording errors
 * during initialization without ara::log.
 *
 * @uptrace{SWS_CORE_10001}
 */
Result<void> Initialize();

/**
 * @brief Destroy all data structures and threads of the AUTOSAR Adaptive Runtime for Applications.
 *
 * @return A Result object that indicates whether the ARA was successfully destroyed.
 * @remark Typical error cases to be
 * reported here are that the user is still holding some
 * resource inside the ARA. Note that this Result is the
 * only way for the ARA to report an error that is
 * guaranteed to be available, e.g., in case ara::log has
 * already been deinitialized. The user is not expected
 * to be able to recover from such an error. However,
 * the user may have a project-specific way of
 * recording errors during deinitialization without
 * ara::log.
 *
 * @uptrace{SWS_CORE_10002}
 */
Result<void> Deinitialize();

}
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_INITIALIZATION_H_
/* EOF */
