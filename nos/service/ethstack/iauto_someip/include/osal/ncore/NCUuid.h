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
 * @file NCUuid.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef UUID_H_
#define UUID_H_
#include "osal/ncore/NCTimeDefine.h"
OSAL_BEGIN_NAMESPACE
typedef UCHAR uuid_st[ 16 ];

struct uuid_data {
    UINT32 time_low;
    UINT16 time_mid;
    UINT16 time_hi_and_version;
    UINT16 clock_seq;
    UINT8  node[ 6 ];
};

VOID uuid_pack_st( const struct uuid_data *uu, uuid_st ptr );
VOID uuid_unpack_st( const uuid_st in, struct uuid_data *uu );

VOID uuid_generate_random_st( uuid_st out );
VOID uuid_generate_time_st( uuid_st out );

VOID uuid_generate_st( uuid_st out );
OSAL_END_NAMESPACE
#endif
