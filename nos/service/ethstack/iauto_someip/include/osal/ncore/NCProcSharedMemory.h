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
 * @file NCProcSharedMemory.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCPROCSHAREDMEMORY_H_
#define INCLUDE_NCORE_NCPROCSHAREDMEMORY_H_

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/**
 *  @brief Processes shared memory
 *
 *  @class NCProcSharedMemory
 *  @note this class does not provide any memory lock between
 */
class __attribute__( ( visibility( "default" ) ) ) NCProcSharedMemory {
   public:
    NCProcSharedMemory();
    virtual ~NCProcSharedMemory();

    /**
     * @brief alloc a named, share memory of size big
     *
     * @param name shared memory name
     * @param size shared memory size
     *
     * @return NULL alloc failed, other case the address of the memory
     *
     */
    UINT8 *mem_alloc( const CHAR *name, const UINT32 size );

    /**
     * @brief alloc a named, share memory of size big
     *
     * @param name shared memory name
     * @param size shared memory size
     *
     * @return NULL calloc failed, other case the address of the memory
     *
     */
    UINT8 *mem_calloc( const CHAR *const name, UINT32 size );

    /**
     * @brief unmapping the memory ,this will delete the memory after all close
     *
     */
    VOID mem_free();

    /**
     * @brief get the alloced size
     *
     * @return the size alloced
     *
     */
    UINT32 mem_getSize() const;

    /**
     * @brief open a existing memory
     *
     * @param name shared memory name
     *
     * @return NULL alloc failed, other case the address of the memory
     *
     */
    UINT8 *mem_open( const CHAR *name );

    /**
     * @brief close the named shared memory, this will unmap the shared memory
     *
     */
    VOID mem_close();

    /**
     *  @brief set memory to zero
     */
    VOID mem_clear();

    /**
     *  @brief read a size large buff from offset
     *
     *  @param offset start piont
     *  @param buff the storing buffer
     *  @param size the length to read
     *
     *  @retval NC_TRUE success
     *  @retval NC_FALSE failed
     */
    NC_BOOL mem_read( const UINT32 offset, UINT8 *const buff, const UINT32 size );

    /**
     *  @brief write a size large buff from offset
     *  @param offset start piont
     *  @param buff the buffer to write
     *  @param size the length to write
     *
     *  @retval NC_TRUE success
     *  @retval NC_FALSE failed
     */
    NC_BOOL mem_write( UINT32 offset, const UINT8 *const data, UINT32 size );

    //        /**
    //         * @brief mapping count
    //         *
    //         * @return memory referenced count
    //         */
    //        inline static UINT32 getTotalSharedMemCount();
    //
    //        /**
    //         * @brief real memory used
    //         *
    //         * @return real memory total size
    //         */
    //        inline static UINT32 getTotalSharedMemSize();

    /**
     * @brief memory address
     *
     * @return shared memory address
     */
    UINT8 *getAddress();

   private:
    VOID                   freeMemory();
    static volatile UINT32 total_sharedmem_count;
    static volatile UINT32 total_sharedmem_size;
    static NCSyncObj       measure_sync;

    INT32   fd;
    UINT8 * addr;
    UINT32  memory_size;
    NC_BOOL created;
    CHAR    memname[ 256 ];

   private:
    NCProcSharedMemory( const NCProcSharedMemory & );
    NCProcSharedMemory &operator=( const NCProcSharedMemory & );
};

//    inline UINT32 NCProcSharedMemory::getTotalSharedMemCount()
//    {
//        return total_sharedmem_count;
//    }
//
//    inline UINT32 NCProcSharedMemory::getTotalSharedMemSize()
//    {
//        return total_sharedmem_size;
//    }
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCPROCSHAREDMEMORY_H_
/* EOF */
