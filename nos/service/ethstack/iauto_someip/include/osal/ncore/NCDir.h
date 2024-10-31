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
 * @file NCDir.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCDIR_H_
#define INCLUDE_NCORE_NCDIR_H_

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
 * @class NCDir
 */
class __attribute__( ( visibility( "default" ) ) ) NCDir {
   public:
    /**
     * @brief Construct a new NCDir object
     *
     * @param bRecLog [IN] Whether record log or not. For NC_TRUE, record log
     */
    explicit NCDir( const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Construct a new NCDir object
     *
     * @param strDirName   The absolute path of the work directory.
     * @param bRecLog      Whether record log or not. For NC_TRUE, record log
     */
    explicit NCDir( const NCString &strDirName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Construct a new NCDir object
     *
     * @param szDirName The absolute path of the work directory.
     * @param bRecLog   Whether record log or not. For NC_TRUE, record log.
     */
    explicit NCDir( const CHAR *const szDirName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Destroy the NCDir object
     *
     */
    ~NCDir();

    /**
     * @brief Set the Work Dir object
     *
     * @param strDirName The absolute path of the work directory.
     * @return VOID
     */
    VOID setWorkDir( const NCString &strDirName );

    /**
     * @brief Flush data of specified dir from IO cache to hardware
     *        not supported for no dirfd API in system.
     *
     * @param strDirPath   need sync dir, must be dir, not file, link or others relative path or
     *                     absolute path
     * @return NC_BOOL     Whether the function call succeed.
     *                     NC_TRUE indicates success.
     *                     Otherwise indicates failure,
     *                     and you can call GetLastError to get detail information.
     *
     * @remarks you can flush different strDirName dirs continuously, eg.
     *  ...
     *  flushDir(dir1);
     *  flushDir(dir2);
     *  ...
     */
    NC_BOOL flushDir( const NCString &strDirPath );

    /**
     * @brief Flush data from IO cache to hardware, you must first set work dir
     *        before flush not supported for no dirfd API in system.
     *
     * @return NC_BOOL  Whether the function call succeed.
     *                  NC_TRUE indicates success.
     *                  Otherwise indicates failure,
     *                  and you can call GetLastError to get detail information.
     *
     * @remarks:
     *      1. usage
     *          1) first setWorkDir(relative or absolute path):
     *          by setWorkDir(const NCString& strDirName) or
     *          NCDir(const NCString& strDirName, const NC_BOOL& bRecLog = NC_TRUE) or
     *          NCDir(const NCCHAR* szDirName, const NC_BOOL& bRecLog = NC_TRUE);
     *
     *          2) then flushDir()
     *      2. if need continuously flush dir, need do as follows:
     *          ...
     *          setWorkDir(dir1);
     *          flushDir();
     *          setWorkDir(dir2);
     *          flushDir();
     *           ...
     */
    NC_BOOL flushDir();

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
     * @brief Move a file or directory to a new place
     *
     * For directory, all items in the old directory will be moved to the new
     directory.
     *
     * @param[in]    strOldName    The name of the file or directory to be moved.
     * @param[in]    strNewName    The name of the file or directory to move to.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.\n
        On Linux, even if the file or directory is used by others, it can be moved
     too.
        So you should ensure that the file or directory is not used when you move
     it. If not, errors will occur.
     */
    NC_BOOL move( const NCString &strOldName, const NCString &strNewName );

    /**
     * @brief Copy a file or directory
     *
     * For directory, all items in the source directory will be copied to the
     destination directory.
     *
     * @param[in]    strSrcName    The name of the file or directory to be copied.
     * @param[in]    strDstName    The name of the file or directory to copy to.
     * @param[in]    isFollowLink   when src is link and dir, copy with follow
     link or not
                        NC_TRUE: copy with follow link
                        NC_FALSE: copy without folllow link

     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.\n
     */
    NC_BOOL copyDir( const NCString &strSrcName, const NCString &strDstName,
                     const NC_BOOL &isFollowLink = NC_FALSE );

    /**
     * @brief Remove a file or directory
     *
     * For directory, all files and subdirectories in the directory will be
     removed.
     *
     * @param[in]    strFileName    The name of the file or directory to be
     removed.
     * @param[in]    bForce        Whether remove the file or directory forcedly
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.\n
        On Linux, even if the file or directory is used by others, it can be
     removed too.
        So you should ensure that the file or directory is not used when you
     remove it. If not, errors will occur.
     */
    NC_BOOL removeDir( const NCString &strFileName, const NC_BOOL &bForce = NC_FALSE );

    /**
     * @brief Create a directory
     *
     * @param[in]    strDirName    The name of the directory to be created.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.\n
        This function can make one level directory only,
        so if the upper level directory does not exist, this function call will
     fail.
     */
    NC_BOOL makeDir( const NCString &strDirName );

    /**
     * @brief Create a path
     *
     * @param[in]    strPathName    The name of the path to be created.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the class instance has work directory, you can use relative or
     absolute path.
        Otherwise, you should use absolute path.\n
        This function can make the full path named in the parameter
     <i>strPathName</i>.
     */
    NC_BOOL makePath( const NCString &strPathName );

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
    UINT32 EntryNum( const UINT32 &dwFilters ) const;

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
            NCDir cDir(NTEXT("your/dir/path"));
            NCDirIterator* pcIter = cDir.EntryIterator(NC_FFT_AllEntries);
            // do something using pcIter
            delete pcIter; // you must do it to avoid memory leak
            // do something else
        \endcode
     */
    NCDirIterator *EntryIterator( const UINT32 &dwFilters );

    /**
     * @brief Get the path of current directory
     *
     * @return The path of current directory.
     * @remarks This function get the path of the directory where the application
     starts
        but not the work directory of the class instance.
     */
    static NCString CurrentDir();

    /**
     * @brief Get the last error code
     *
     * @return The last error code. See \ref NCFileError.
     */
    NCFileError getLastError() const;

    /**
     * @brief Get space usage of the file system which current directory is on in
     kb.
     *
     * @param[out]  dwAvail  Free size available to unprivileged user.
     * @param[out]  dwFree   Free size in current file system.
     * @param[out]  dwTotal  Total size in current file system.
     * @return Whether get space usage of the file system size succeed. NC_TRUE
     indicates success.
        Otherwise indicates failure.
     */
    NC_BOOL getVolumeSize( UINT64 *dwAvail, UINT64 *dwFree, UINT64 *dwTotal ) const;

   private:
    NCDirImpl *m_ptr;

   private:
    NCDir( const NCDir & );
    NCDir &operator=( const NCDir & );
};

/**
 * @brief Entry iterator of a directory
 *
 * @class NCDirIterator
 *
 * @par Usage sample:
    \code
        NCDir cDir(NTEXT("your/dir/path"));
        NCDirIterator* pcIter = cDir.EntryIterator(NCFFT_AllEntries);
        while (!pcIter->End())
        {
            NCString strFileName = pcIter->CurrentFileName();
            NCFileInfo cFileInfo = pcIter->CurrentFileInfo();

            // do something else

            ++(*pcIter);
        }
        delete pcIter;    // you must do it to avoid memory leak
    \endcode
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCDirIterator {
   public:
    friend class NCDirImpl;
    /**
     * @brief Destructor
     */
    ~NCDirIterator();

    /**
     * @brief Get the name of current entry
     *
     * @return The name of current entry.
     */
    NCString CurrentFileName() const;

    /**
     * @brief Get the file info of current entry
     *
     * @return The file info(see \ref NCFileInfo) of current entry.
     */
    NCFileInfo CurrentFileInfo() const;

    /*
     * @brief Get the info of current entry
     */
    // NCFileInfo CurrentFileInfo();

    /**
     * @brief Check whether there are more entries
     *
     * @return NC_TRUE indicates that there are more entries. Otherwise indicates
     * no more entry.
     */
    NC_BOOL End() const;

    /**
     * @brief Move to next entry
     *
     * This function overwrites the left-increment operator
     */
    NCDirIterator &operator++();

    /*
     * @brief Move to next entry
     */
    // const NCDirIterator operator++(INT);

    /*
     * @brief Get last error code
     */
    // NCFileError GetLastError() const;

    /**
     * @brief Reset the iterator
     *
     * After this function call, the iterator points to the first entry
     */
    VOID reset();

   private:
    /**
     * @brief Constructor
     *
     * @param[in]    strDirPath    The directory whose entries to iterate.
     * @param[in]    dwFilters    Entry filters. See \ref NCFileFilter.
     * @param[in]    bRecLog        Whether record log or not. For NC_TRUE, record
     * log.
     */
    NCDirIterator( const NCString &strDirPath, const UINT32 &dwFilters, const NC_BOOL &bRecLog );
    NCDirIteratorImpl *m_ptr;

   private:
    NCDirIterator( const NCDirIterator & );
    NCDirIterator &operator=( const NCDirIterator & );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCDIR_H_
/* EOF */
