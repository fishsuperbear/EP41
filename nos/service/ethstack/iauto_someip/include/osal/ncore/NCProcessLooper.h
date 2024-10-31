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
 * @file NCProcessLooper.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCPROCESSLOOPER_H
#define NCPROCESSLOOPER_H
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCRunnableLooper.h"

#if defined( __ANDROID__ )
extern __attribute__( ( visibility( "default" ) ) ) OSAL::NCRunnableLooper GetMainLooper();
#else
extern "C" {
extern __attribute__( ( visibility( "default" ) ) ) OSAL::NCRunnableLooper GetMainLooper();
}
#endif

#endif  // NCPROCESSLOOPER_H
/* EOF */
