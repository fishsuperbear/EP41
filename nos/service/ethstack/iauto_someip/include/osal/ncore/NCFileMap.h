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
 * @file NCFileMap.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCFILEMAP_H
#define NCFILEMAP_H

#include <sys/mman.h>

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include "osal/ncore/NCString.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE

static VOID *const NC_INVALID_ADDRESS = static_cast<VOID *>( MAP_FAILED );

// class declaration
class NCFileMap;
class NCFileMapImpl;

/**
 * @brief mapping file into memory
 *
 * @class NCFileMap
 */
class __attribute__( ( visibility( "default" ) ) ) NCFileMap {
   public:
    /**
     * @brief Construct a new NCFileMap
     *
     */
    NCFileMap();

    /**
     *  @brief Construct of NCFileMap
     *  @param strFileName : the name of the file to be operator on
     *
     */
    explicit NCFileMap( const NCString &strFileName );

    /**
     * @brief Destruct of NCFileMap
     *
     */
    virtual ~NCFileMap();

    /**
     *  @brief map named file into memory
     *
     *  @param strFileName : the name of file to be mapped
     *  @param filesize : the filesize to be mapped, if file already exist, this
     * size will not used, the whole file will be mapped
     *  @param prot : the memory permission;
     *  - NC_FM_PROT_READ : Pages may be read.
     *  - NC_FM_PROT_WRITE : Pages may be write.
     *  - NC_FM_PROT_EXEC : Pages may be executed.
     *  - NC_FM_PROT_NONE : Pages may not be accessed.
     *  @param flags : while could be shared or apl private;
     *  - NC_FM_FLAGS_PRIVATE : Create a private copy-on-write mapping, write
     * operation is not visible to other process ,also content it write will not
     * write to the disk
     *  - NC_FM_FLAGS_SHARED : Share this mapping, visible to other process
     *  @return the address of file mapping memory
     *          NC_INVALID_ADDRESS : mapping failed
     *  @note
     *  - if file does not exist, this call will return NC_INVALID_ADDRESS
     *  - this call will close the last address it already mapped
     */
    VOID *mapVirtualMemory( const NCString &strFileName, const UINT32 filesize, UINT32 prot,
                            UINT32 flags );

    /**
     *  @brief map named file into memory
     *
     *  @param strFileName : the name of file to be mapped
     *  @param prot : the memory permission
     *  - NC_FM_PROT_READ : Pages may be read.
     *  - NC_FM_PROT_WRITE : Pages may be write.
     *  - NC_FM_PROT_EXEC : Pages may be executed.
     *  - NC_FM_PROT_NONE : Pages may not be accessed.
     *  @param flags : while could be shared or apl private
     *  - NC_FM_FLAGS_PRIVATE : Create a private copy-on-write mapping, write
     * operation is not visible to other process ,also content it write will not
     * write to the disk
     *  - NC_FM_FLAGS_SHARED : Share this mapping, visible to other process
     *  @return the address of file mapping memory
     *          NC_INVALID_ADDRESS : mapping failed
     *
     *  @note
     *  - if file does not exist, this call will return NC_INVALID_ADDRESS
     *  - this call will close the last address it already mapped
     */
    VOID *mapVirtualMemory( const NCString &strFileName, UINT32 prot, UINT32 flags );

    /**
     *  @brief map file into memory
     *
     *  @param prot : the memory permission
     *  - NC_FM_PROT_READ : Pages may be read.
     *  - NC_FM_PROT_WRITE : Pages may be write.
     *  - NC_FM_PROT_EXEC : Pages may be executed.
     *  - NC_FM_PROT_NONE : Pages may not be accessed.
     *  @param flags : while could be shared or apl private
     *  - NC_FM_FLAGS_PRIVATE : Create a private copy-on-write mapping, write
     * operation is not visible to other process ,also content it write will not
     * write to the disk
     *  - NC_FM_FLAGS_SHARED : Share this mapping, visible to other process
     *  @return the address of file mapping memory
     *          NC_INVALID_ADDRESS : mapping failed
     *
     *  @note
     *  - if file does not exist, this call will return NC_INVALID_ADDRESS
     *  - this call will close the last address it already mapped
     *
     */
    VOID *mapVirtualMemory( UINT32 prot, UINT32 flags );

    /**
     * @brief unmap file
     *
     * @param address : address will be unmapped
     * @return NC_FALSE : unmapping failed
     *         NC_TRUE : unmapping success
     * @note
     * if you forget unmap after mmap , this will cause a memory leak
     *
     */
    NC_BOOL unmapVirtualMemory();

    /**
     * @brief get the virtual memory address of the file mapping
     *
     * @return the address of file mapping memory
     *          NC_INVALID_ADDRESS : map does not exist
     */
    VOID *getMapAddress();

    /**
     * @brief get the virtual memory address of the file mapping
     *
     * @return the size had been mapped
     */
    UINT32 getMapSize() const;

    /**
     * @brief write back to disk immediately
     *
     * @return NC_FALSE : sync failed
     *         NC_TRUE : sync success
     */
    NC_BOOL flushMap();

   private:
    NCFileMapImpl *m_ptr;
};
OSAL_END_NAMESPACE
#endif /* NCFILEMAP_H */
/* EOF */
