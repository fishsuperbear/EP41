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
 * @file NCTmpFile.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTMPFILE_H_
#define INCLUDE_NCORE_NCTMPFILE_H_

#include "osal/ncore/NCFileInfo.h"
#include "osal/ncore/NCFilePubDef.h"
#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// external class declaration
class NCFileImpl;

/**
 * @brief
 *
 * @class NCTmpFile
 * In RAM filesystem, don't supoort APIs: fsync, flock, statvfs, statvfs64,
 * mkdir, symlink, readlink.
 */
class __attribute__( ( visibility( "default" ) ) ) NCTmpFile {
   public:
    /**
     * @brief Constructor
     *
     * @param[in]    bRecLog    Whether record log or not. For NC_TRUE, record
     * log.
     * @note If you use this constructor, you should call \ref SetFileName before
     * you manipulate a file.
     */
    explicit NCTmpFile( const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief \overload
     *
     * @param[in]    strFileName    The absolute path of the file.
     * @param[in]    bRecLog        Whether record log or not. For NC_TRUE, record
     log.
     * @remarks Before you call \ref SetFileName to change the file to manipulate,
        all operations will act on the file specified by the parameter
     <i>strFileName</i>.
     */
    explicit NCTmpFile( const NCString &strFileName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief \overload
     *
     * @param[in]    szFileName    The absolute path of the file.
     * @param[in]    bRecLog        Whether record log or not. For NC_TRUE, record
     log.
     * @remarks Before you call \ref SetFileName to change the file to manipulate,
        all operations will act on the file specified by the parameter
     <i>szFileName</i>.
     */
    explicit NCTmpFile( const CHAR *const szFileName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Destructor
     */
    virtual ~NCTmpFile();

    /**
     * @brief Set the name of the file to manipulate
     *
     * @param[in]    strFileName    The absolute path of the file.
     * @remarks After this function call, all operations will be changed to the
     file
        specified by the parameter <i>strFileName</i>.
     */
    VOID setFileName( const NCString &strFileName );

    /**
     * @brief Open the file
     *
     * @param[in]    iOpenMode    The open mode. See \ref NCFileOpenMode.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks After open a file, you should close it using \ref Close.
     */
    virtual NC_BOOL openFile( const UINT32 &iOpenMode );

    /**
     * @brief Close the file
     *
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks After open a file using \ref Open, you should close it.
     */
    virtual NC_BOOL closeFile();

    /**
     * @brief Check whether the file is open
     *
     * @return NC_TRUE indicates that the file is open. Otherwise indicates the
     * file is not open.
     */
    virtual NC_BOOL isOpen() const;

    /**
     * @brief Read data from the file
     *
     * @param[in]    pBuff        The pointer to the buffer that accepts the data
     read from the file.
     * @param[in]    dwMaxSize    The maximum size to read from the file.
     * @return The return value indicates the size that actually read from the
     file.
        If reaching the end of file(EOF) or some error occurs, the return value is
     zero.
        So you should use \ref AtEnd to check whether reaches the end of file.
        If the return value is zero and \ref AtEnd returns NC_TRUE, that means
     reaching the end of file.
        Otherwise means some error occurs, and you can call \ref GetLastError to
     get detail information.
     */
    virtual UINT32 readFile( VOID *const pBuff, const UINT32 &dwMaxSize );

    /**
     * @brief Write data to the file
     *
     * @param[in]    pBuff    The pointer to the buffer that contains the data
     write to the file.
     * @param[in]    dwSize    The data size to write to the file.
     * @return The return value indicates the size that actually write to the
     file.
        If some error occurs, the return value is zero, and you can call \ref
     GetLastError to get detail information.
     */
    virtual UINT32 writeFile( const VOID *pBuff, const UINT32 &dwSize );

    /**
     * @brief Move the file pointer.
     *
     * The function moves the position of file pointer for reading or writing.
     *
     * @param[in]    offset    The offset to move from the starting point.
                        If its value is negative, then move backward; if its value
     is positive, then move foreward.
     * @param[in]    eMode    Specify the starting point. See \ref NCFileSeekMode.
     * @return The return value indicates the new position of the file pointer(the
     offset from the beginning of the file).
        If some error occurs, the return value is -1, and you can call \ref
     GetLastError to get detail information.
     */
    virtual INT64 seek( const INT64 &offset, const NCFileSeekMode &eMode );

    /**
     * @brief Check whether reach the end of file
     *
     * @return NC_TRUE indicates reaching the end of file. Otherwise indicates
     * not.
     */
    virtual NC_BOOL AtEnd();

    /**
     * @brief Check whether the file exists
     *
     * @return NC_TRUE indicates that the file or directory exists. Otherwise
     * indicates not exist.
     */
    NC_BOOL exists() const;

    /**
     * @brief Check whether a file exists
     *
     * @param[in]    strFileName    The absolute path of the file to check.
     * @return NC_TRUE indicates that the file or directory exists. Otherwise
     * indicates not exist.
     */
    static NC_BOOL exists( const NCString &strFileName );

    /**
     * @brief Remove the file
     *
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks On Linux, even if the file or directory is used by others, it can
     be removed too.
        So you should ensure that the file or directory is not used when you
     remove it. If not, errors will occur.
     */
    NC_BOOL removeFile();

    /**
     * @brief Remove a file
     *
     * @param[in]    strFileName    The absolute path of the file to remove.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     otherwise indicates failure.
     * @remarks On Linux, even if the file or directory is used by others, it can
     be removed too.
        So you should ensure that the file or directory is not used when you
     remove it. If not, errors will occur.
     */
    static NC_BOOL removeFile( const NCString &strFileName );

    /**
     * @brief Move the file
     *
     * This function move the file to the new place or just rename it.
     *
     * @param[in]     strNewName    The absolute path of the new file.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks On Linux, even if the file or directory is used by others, it can
     be moved too.
        So you should ensure that the file or directory is not used when you move
     it. If not, errors will occur.\n
        If the new file has already existed, this function fails and the file is
     not moved to the new place.
     */
    NC_BOOL moveto( const NCString &strNewName );

    /**
     * @brief Move a file
     *
     * This function move a file from the old place to a new place or just rename
     it.
     *
     * @param[in]    strOldName    The absolute path of the old file.
     * @param[in]    strNewName    The absolute path of the new file.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     otherwise indicates failure.
     * @remarks On Linux, even if the file or directory is used by others, it can
     be moved too.
        So you should ensure that the file or directory is not used when you
     remove it. If not, errors will occur.\n
        If the new file has already existed, this function fails and the file is
     not moved to the new place.
     */
    static NC_BOOL move( const NCString &strOldName, const NCString &strNewName );

    /**
     * @brief Copy the file
     *
     * This function copy the file to the destination place.
     *
     * @param[in]    strDstFileName    The absolute path of the destination file.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     * @remarks If the destination file has already existed, this function fails
     and the file is not copied
        and the destination file is not be overwritten.
     */
    NC_BOOL copyto( const NCString &strDstFileName );

    /**
     * @brief Copy a file
     *
     * This function copy the source file to the destination place.
     *
     * @param[in]    strSrcFileName    The absolute path of the source file.
     * @param[in]    strDstFileName    The absolute path of the destination file.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     otherwise indicates failure.
     * @remarks If the destination file has already existed, this function fails
     and the file is not copied
        and the destination file is not be overwritten.
     */
    static NC_BOOL copyFile( const NCString &strSrcFileName, const NCString &strDstFileName );

    // LONG Size() const;
    // static LONG Size(const NCString& strFileName);

    /**
     * @brief Set the size of the file
     *
     * This function changes the size of a file. If the size is less than the
     original size of the file,
     * the file will be truncated. Otherwise, the file will be appended with zero.
     *
     * @param[in]    size    The size of the file.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     */
    NC_BOOL setSize( const INT64 &lSize );

    /**
     * @brief Set the size of a file
     *
     * This function changes the size of a file. If the size is less than the
     * original size of the file,
     * the file will be truncated. Otherwise, the file will be appended with zero.
     *
     * @param[in]    strFileName    The absolute path of the file.
     * @param[in]    size        The size of the file.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     * otherwise indicates failure.
     */
    static NC_BOOL setSize( const NCString &strFileName, const INT64 &size );

    /**
     * @brief Set the permissions of the file
     *
     * @param[in]    dwPerms    The file permissions. See \ref NCFilePermission.
     * @return Whether the function call succeed. NC_TRUE indicates success.
        Otherwise indicates failure, and you can call \ref GetLastError to get
     detail information.
     */
    NC_BOOL setPermissions( const UINT32 &dwPerms );

    /**
     * @brief Set the permissions of a file
     *
     * @param[in]    strFileName    The absolute path of the file.
     * @param[in]    dwPerms        The file permissions. See \ref
     * NCFilePermission.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     * otherwise indicates failure.
     */
    static NC_BOOL setPermissions( const NCString &strFileName, const UINT32 &dwPerms );

    /**
     * @brief Get the permissions of a file
     *
     * @return the \ref NCFilePermission of a file or directory
     */
    UINT32 permissions();

    /**
     * @brief Get the permissions of a file
     *
     * @param[in]    strFileName    The absolute path of the file.
     * @return the \ref NCfilePermission of a file or directory
     */
    static UINT32 permissions( const NCString &strFileName );

    /**
     * @brief Set end of a file, truncate file into current size
     *
     * @return Whether the function call succeed. NC_TRUE indicates success,
     * otherwise indicates failure.
     */
    NC_BOOL setEndOfFile();

    /**
     * @brief Get the file info of the file
     *
     * @return The file info of the file. See \ref NCFileInfo.
     */
    NCFileInfo FileInfo() const;

    /**
     * @brief Get the last error code
     *
     * @return The last error code. See \ref NCFileError.
     */
    NCFileError getLastError() const;

   protected:
    explicit NCTmpFile( NCFileImpl *const ptr );

   private:
    NCTmpFile( const NCTmpFile & );
    NCTmpFile &operator=( const NCTmpFile & );

    NCFileImpl *m_ptr;
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTMPFILE_H_
/* EOF */
