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
 * @file NCMd5.h
 * @brief
 * @date 2020-08-20
 *
 */

#ifndef NCMD5_H
#define NCMD5_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE

#define MD5_DIGEST_LENGTH_L 16

class Md5 {
   public:
    /**
     * @brief get the MD5 Handler
     *
     * @return char* MD5 Handler
     */
    static char* NCMd5Handler();

    /**
     * @brief [IN ]free the MD5 Handler
     *
     * @param handler MD5 Handler
     */
    static void NCMd5HandlerFree( char* handler );

    /**
     * @brief [OUT]init stru
     *
     * @param stru the data want to init
     */
    static void NCMD5Init( char* stru );

    /**
     * @brief update buf to stru
     *
     * @param stru  [OUT] the data wait to update
     * @param buf   [IN]  the buffer to update stru
     * @param len   [IN]  the length of buf
     */
    static void NCMD5Update( char* stru, unsigned char* buf, unsigned int len );

    /**
     * @brief update digest with stru
     *
     * @param digest
     * @param stru
     */
    static void NCMD5Final( unsigned char digest[ MD5_DIGEST_LENGTH_L ], char* stru );
};
OSAL_END_NAMESPACE
#endif /* NCMD5_H */
/* EOF */
