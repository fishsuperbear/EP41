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

#ifndef APD_ARA_CORE_INITIALIZATION_IMPL_H_
#define APD_ARA_CORE_INITIALIZATION_IMPL_H_

#include <functional>

#include "ara/core/array.h"
#include "ara/core/result.h"
#include "ara/core/vector.h"

namespace ara {
namespace core {
namespace Initialization {
inline namespace _19_11 {
enum class InitializationLevel : uint8_t {
    Level0 = 0U,
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
    LevelMax
};
const InitializationLevel LEV_LOG     = InitializationLevel::Level1;
const InitializationLevel LEV_DEFAULT = InitializationLevel::Level3;
const uint8_t             LEV_MAX     = static_cast<uint8_t>( InitializationLevel::LevelMax );

/**
 * @brief the implement of initialization
 */
class InitializationImpl {
   public:
    /**
     * @brief get the instance of InitializationImpl
     *
     * @return InitializationImpl* the instance of InitializationImpl
     */
    static InitializationImpl* Ins();

    /**
     * @brief register the initialization callback function
     *
     * @param callback the initialization callback function
     */
    void RegeditInitializeCallBack( std::function<Result<void>()> callback,
                                    InitializationLevel           lev = LEV_DEFAULT );

    /**
     * @brief register the deinitialization callback function
     *
     * @param callback the deinitialization callback function
     */
    void RegeditDeInitializeCallBack( std::function<Result<void>()> callback,
                                      InitializationLevel           lev = LEV_DEFAULT );

    /**
     * @brief initialization function
     *
     * @return Result<void> the result of initialization
     */
    Result<void> Initialize();

    /**
     * @brief deinitialization function
     *
     * @return Result<void> Result<void> the result of deinitialization
     */
    Result<void> Deinitialize();

    /**
     * @brief determine whether the initialization is complete
     *
     * @return true     initialize finish
     *         false    not initialize
     */
    bool HasInitialize();

   private:
    /**
     * @brief Construct a new Initialization Impl
     */
    InitializationImpl();

    /**
     * @brief Destroy the Initialization Impl
     */
    ~InitializationImpl();

   private:
    Array<Vector<std::function<Result<void>()>>, LEV_MAX> mInitializeCallbackList;
    Array<Vector<std::function<Result<void>()>>, LEV_MAX> mDeInitializeCallbackList;
    bool                                                  mHasInitialize;
};
}  // namespace _19_11
}  // namespace Initialization
}  // namespace core
}  // namespace ara

#endif  // APD_ARA_CORE_INITIALIZATION_IMPL_H_
/* EOF */
