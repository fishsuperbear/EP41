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
 * @file NCThreadErrorNum.h
 * @brief
 * @date 2021-03-02
 * @author sunbin
 */

#ifndef INCLUDE_NCORE_NCTHREADERRORNUM_H_
#define INCLUDE_NCORE_NCTHREADERRORNUM_H_

#include <thread>

#include "NCNameSpace.h"
#include "NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

class NCThreadErrorNum {

   public:
    NCThreadErrorNum()  = default;
    ~NCThreadErrorNum() = default;

    static UINT32 get();
    static VOID   set( UINT32 err );

   private:
    NCThreadErrorNum( const NCThreadErrorNum & ) = delete;
    NCThreadErrorNum &operator=( const NCThreadErrorNum & ) = delete;
};

OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCSYNCOBJ_H_
/* EOF */