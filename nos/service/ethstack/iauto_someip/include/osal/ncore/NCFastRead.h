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
 * @file NCFastRead.h
 * @brief
 * @date 2020-07-23
 *
 */

#ifndef INCLUDE_NCORE_NCFASTREAD_H
#define INCLUDE_NCORE_NCFASTREAD_H

#include "osal/ncore/NCFile.h"
#include "osal/ncore/NCProcSharedMemory.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
class NCFastRead {
   public:
    /**
     * @brief Construct a new NCFastRead object
     *
     * @param fileName file name. eg: machine_manifest.json
     */
    NCFastRead( const NCString& fileName );
    ~NCFastRead();

    NCFastRead( const NCFastRead& other ) = delete;
    NCFastRead& operator=( const NCFastRead& other ) = delete;

    /**
     * @brief get file data with shared memory
     *
     * @param [OUT] fileSize the size of data
     * @return CHAR*  the pointer of data
     */
    CHAR* getDataFromShared( UINT32& fileSize );

   private:
    NCString m_fileName;
    UINT32   m_fileSize;
    CHAR*    m_paddr;
};
OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCFASTREAD_H