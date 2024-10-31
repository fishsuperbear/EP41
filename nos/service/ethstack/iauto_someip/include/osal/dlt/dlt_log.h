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
 * @file dlt_log.h
 * @brief
 * @date 2020-06-18
 *
 */
#ifndef DLT_LOG_H
#define DLT_LOG_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <stdint.h>
#include <stdlib.h>

#include <functional>
#include <string>

#include "osal/dlt/dlt_common.h"
#include "osal/dlt/dlt_conf.h"
#include "osal/dlt/dlt_payload.h"
#include "osal/dlt/dlt_protocol.h"

namespace dlt {

class DltLogItemPacker {
   public:
    DltLogItemPacker();
    virtual ~DltLogItemPacker();

    static bool RegeditTSyncCB( std::function<time_t( void )> func );

    template <typename T>
    uint32_t PackMsgPayload( const T v, DltFormatType type = DLT_FORMAT_DEFAULT ) {
        return mPayload->PackMsgPayload<T>( v, type );
    };

    template <typename T = char *>
    uint32_t PackMsgPayload( const char *array, uint16_t len ) {
        return mPayload->PackMsgPayload<T>( array, len );
    }

    template <typename T = void *>
    uint32_t PackMsgPayload( const void *array, uint16_t len ) {
        return mPayload->PackMsgPayload<T>( array, len );
    }

    virtual uint32_t PackEnd() = 0;
    virtual void     Reset()   = 0;

    virtual const uint8_t *GetBuffer() const     = 0;
    virtual uint32_t       GetBufferSize() const = 0;

   protected:
    time_t OnTsync();

   protected:
    DltPayload *mPayload;

   private:
    static std::function<time_t( void )> gDltOnTSync;
};
}  // namespace dlt
#endif /* __DLT_LOG_H__ */