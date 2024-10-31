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
 * @file NCSharedData.h
 * @brief
 * @date 2020-06-03
 *
 */

#ifndef NCSHAREDDATA_H_
#define NCSHAREDDATA_H_

#include <string>

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/**
 * @brief
 *
 * @class NCSharedData
 */
class __attribute__( ( visibility( "default" ) ) ) NCSharedData {
   public:
    NCSharedData();
    NCSharedData( const std::string& data );
    NCSharedData( const VOID* data, size_t size );

    virtual ~NCSharedData();

    const std::string* object() const;
    const CHAR*        data() const;
    size_t             dataSize();
    INT32              getData( VOID* buf, size_t buf_len );

    NCSharedData( const NCSharedData& other );
    NCSharedData& operator=( const NCSharedData& other );
    NC_BOOL       operator==( const NCSharedData& other );
    NC_BOOL       operator!=( const NCSharedData& other );

    VOID clearData();
    VOID setData( const std::string& data );
    VOID setData( const VOID* data, size_t size );
    VOID appendData( const std::string& data );
    VOID appendData( const VOID* data, size_t size );

   public:
    std::string m_data;
};

OSAL_END_NAMESPACE
#endif /* NCSHAREDDATA_H_ */
       /* EOF */