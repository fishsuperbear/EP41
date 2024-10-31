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
 * @file NCTmpDir.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTMPDIR_H_
#define INCLUDE_NCORE_NCTMPDIR_H_

#include "osal/ncore/NCFileInfo.h"
#include "osal/ncore/NCFilePubDef.h"
#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// external class declaration
class NCDirImpl;
class NCDirIteratorImpl;

class NCDirIterator;

/**
 * @brief
 *
 * @class NCTmpDir
 * In RAM filesystem, don't supoort APIs: fsync, flock, statvfs, statvfs64,
 * mkdir, symlink, readlink.
 */
class __attribute__( ( visibility( "default" ) ) ) NCTmpDir {
   public:
    // NCTmpDir();
    /**
     * @brief Constructor
     *
     * @param[in]    bRecLog    Whether record log or not. For NC_TRUE, record
     * log.
     */
    explicit NCTmpDir( const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief \overload
     *
     * @param[in]    strDirName    The absolute path of the work directory.
     * @param[in]    bRecLog        Whether record log or not. For NC_TRUE, record
     * log.
     */
    explicit NCTmpDir( const NCString &strDirName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief \overload
     *
     * @param[in]    szDirName    The absolute path of the work directory.
     * @param[in]    bRecLog        Whether record log or not. For NC_TRUE, record
     * log.
     */
    explicit NCTmpDir( const CHAR *const szDirName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Destructor
     */
    ~NCTmpDir();

    /**
     * @brief Set the work directory
     *
     * @param[in]    strDirName    The absolute path of the work directory.
     */
    VOID setWorkDir( const NCString &strDirName );

    /**
     * @brief Check whether a file or directory exists
     *
     * @param[in]    strFileName    The name of the file or directory to check.
     * @return NC_TRUE indicates that the file or directory exists. Otherwise
     indicates not exist.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.
     */
    NC_BOOL exists( const NCString &strFileName );

    /**
     * @brief Check whether current work directory exists
     *
     * @return NC_TRUE indicates current directory exists. Otherwise indicates not
     * exist.
     */
    NC_BOOL exists();

    /**
     * @brief Get the number of entries
     *
     * @return The number of entries in the directory. If there's no entry or
     * error occur, zero is return.
     * @remarks "." and ".." would not be included into the final count number.
     *
     */
    UINT32 EntryNum();

    /**
     * @brief Get the number of entries
     *
     * @param[in]    dwFilters    Entry filters. See \ref NCFileFilter.
     * @return The number of entries in the directory. If there's no entry or
     * error occur, zero is return.
     * @remarks if NC_FFT_NoDotAndDotDot was set in dwFilters, "." and ".." would
     * not be included.
     * Otherwise, the number of entries would include "." and "..".
     *
     */
    UINT32 EntryNum( const UINT32 &dwFilters );

    /**
     * @brief Get the entry iterator
     *
     * @param[in]    dwFilters    Entry filters. See \ref NC_FileFilter.
     * @return The pointer to the entry iterator. See more about \ref
     NC_DirIterator.
     * @remarks To use this function, the class instance should have work
     directory.
        Without the work directory, you can use \ref SetWorkDir to set the work
     directory first.\n
        After using the entry iterator, you should delete it.
        There are some sample code:
        \code
            NCTmpDir cDir(NTEXT("your/dir/path"));
            NCDirIterator* pcIter = cDir.EntryIterator(NC_FFT_AllEntries);
            // do something using pcIter
            delete pcIter; // you must do it to avoid memory leak
            // do something else
        \endcode
     */
    NCDirIterator *EntryIterator( const UINT32 &dwFilters ) const;

    /**
     * @brief Get the last error code
     *
     * @return The last error code. See \ref NCFileError.
     */
    NCFileError getLastError() const;

   private:
    NCDirImpl *m_ptr;

   private:
    NCTmpDir( const NCTmpDir & );
    NCTmpDir &operator=( const NCTmpDir & );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTMPDIR_H_
/* EOF */
