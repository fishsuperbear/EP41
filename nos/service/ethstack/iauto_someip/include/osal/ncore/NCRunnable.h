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
 * @file NCRunnable.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCRUNNABLE_H_
#define INCLUDE_NCORE_NCRUNNABLE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCRefBase.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE
/**
 * @class NCRunnable
 *
 * @brief Class for NCRunnable
 * **/
class __attribute__( ( visibility( "default" ) ) ) NCRunnable {
   public:
    /**
     * @brief Construct a new NCRunnable object
     */
    NCRunnable();

    /**
     * @brief Destroy the NCRunnable object
     */
    virtual ~NCRunnable();

    /**
     * @brief run
     */
    virtual void run() = 0;

    /**
     * @brief dump
     *
     * @param out_ out string
     */
    virtual void dump( NCString &out_ );
};

typedef ncsp<NCRunnable>::sp NCRunnableHolder;
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCRUNNABLE_H_
/* EOF */
