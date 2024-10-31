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
 * @file NCFileInfo.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCFILEINFO_H
#define NCFILEINFO_H

#include "osal/ncore/NCFilePubDef.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCTime.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// external class declaration
class NCFileInfoImpl;

/**
 * @brief
 *
 * @class NCFileInfo
 */
class __attribute__( ( visibility( "default" ) ) ) NCFileInfo {
   public:
    /**
     * @brief Constructor
     *
     * @param[in]    bRecLog    Whether record log or not. For NC_TRUE, record
     * log.
     * @note If you use this constructor, you should call \ref SetFileName before
     * you get info of a file.
     */
    explicit NCFileInfo( const NC_BOOL &bRecLog = NC_TRUE );

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
    explicit NCFileInfo( const NCString &strFileName, const NC_BOOL &bRecLog = NC_TRUE );

    /**
     * @brief Copy constructor
     *
     * @param[in]    cInfo    Another instance of \ref NCFileInfo.
     * @remarks Before you call \ref SetFileName to change the file to manipulate,
        all operations will act on the file specified by the file name of
     <i>cInfo</i>.
     */
    NCFileInfo( const NCFileInfo &cInfo );

    /**
     * @brief Copy function
     *
     * @param[in]    cInfo    Another instance of \ref NCFileInfo.
     * @return The reference to this instance.
     * @remarks After this function, the operation of this instance will be
     * changed to the file name of <i>cInfo</i>.
     */
    NCFileInfo &operator=( const NCFileInfo &cInfo );

    /**
     * @brief Destructor
     */
    virtual ~NCFileInfo();

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
     * @brief Get the file name
     *
     * @return The file path that this instance manipulate.
     */
    NCString FileName() const;

    /**
     * @brief check file exist
     *
     * @retval NC_TRUE file exist
     * @retval NC_FALSE file not exist
     */
    NC_BOOL exists() const;

    /**
     * @brief Check whether the file is readable
     *
     * @return NC_TRUE indicates that the file is readable, otherwise the file is
     * not readable.
     */
    NC_BOOL isReadable() const;

    /**
     * @brief Check whether the file is writable
     *
     * @return NC_TRUE indicates that the file is writable, otherwise the file is
     * not writable.
     */
    NC_BOOL isWritable() const;

    /**
     * @brief Check whether the file is executable
     *
     * @return NC_TRUE indicates that the file is executable, otherwise the file
     * is not executable.
     */
    NC_BOOL isExecutable() const;

    /**
     * @brief Check whether the file is hidden
     *
     * @return NC_TRUE indicates that the file is hidden, otherwise the file is
     * not hidden.
     */
    NC_BOOL isHidden() const;

    /**
     * @brief get the file type
     *
     * @retval NC_FFG_TypeFile      Normal file
     * @retval NC_FFG_TypeDirectory Directory
     * @retval NC_FFG_TypeLink      Link file
     * @retval NC_FFG_TypeDevice    Device
     * @retval NC_FFG_TypePipe      Pipe file
     * @retval NC_FFG_TypeSocket    Socket
     */
    UINT32 getFileType() const;

    /**
     * @brief Check whether the file is a normal file
     *
     * @return NC_TRUE indicates that the file is normal file, otherwise the file
     * is not normal file.
     */
    NC_BOOL isFile() const;

    /**
     * @brief Check whether the file is a directory
     *
     * @return NC_TRUE indicates that the file is a directory, otherwise the file
     * is not a directory.
     */
    NC_BOOL isDir() const;

    /**
     * @brief Check whether the file is a link file
     *
     * @return NC_TRUE indicates that the file is a link file, otherwise the file
     * is not a link file.
     */
    NC_BOOL isLink() const;

    // NCString LinkTarget() const;

    /**
     * @brief Get the creation time of the file
     *
     * @return The creation time of the file.
     */
    NCTime CreationTime() const;

    /**
     * @brief Get the last-write time of the file
     *
     * @return The last-write time of the file.
     */
    NCTime LastWriteTime() const;

    /**
     * @brief Get the last-access time of the file
     *
     * @return The last-access time of the file.
     */
    NCTime LastAccessTime() const;

    /**
     * @brief Get the size of the file
     *
     * @return The size of the file. If the function call fails, the return value
     is -1,
        and you can call \ref GetLastError to get detail information.
     */
    INT64 Size() const;

    /**
     * @brief Get the number of entries in the directory
     *
     * @param[in]    dwFilters    Entry filters. See \ref NC_FileFilter.
     * @return The number of entries in the directory. If there's no entry or
     * error occur, zero is return.
     */
    UINT32 EntryNum( const UINT32 &dwFilters ) const;

    /**
     * @brief Get the last error code
     *
     * @return The last error code. See \ref NCFileError.
     */
    NCFileError getLastError() const;

    /**
     * @brief get file path
     *
     * @return the file name, including the path (which may be absolute or
     * relative).
     *
     */
    NCString filePath() const;

   private:
    NCFileInfoImpl *m_ptr;
    NC_BOOL         m_bRecLog;
};
OSAL_END_NAMESPACE

#endif /* NCFILEINFO_H */
/* EOF */
