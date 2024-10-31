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
 * @file NCPath.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCPATH_H
#define NCPATH_H

#include "osal/ncore/NCString.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// external class declaration
class NCPathImpl;

/**
 * @brief The class to process paths
 *
 * @class NCPath
 */
class __attribute__( ( visibility( "default" ) ) ) NCPath {
   public:
    /**
     * @brief Constructor
     *
     * @param[in]    strPathName    The path to be processed.
     */
    explicit NCPath( const NCString &strPathName );

    /**
     * @brief Destructor
     */
    ~NCPath();

    /**
     * @brief Path separator
     *
     * @return Path separator according to the native platform.
     */
    static CHAR Separator();

    /**
     * @brief Convert the path to native platform
     *
     * @return Whether the function call succeed. NC_TRUE indicates success,
     * otherwise indicates failure.
     */
    NC_BOOL Native();

    /**
     * @brief Convert the path to absolute path
     *
     * If the path is relative, this function can convert it to absolute path
     * according to the current directory you specified.
     *
     * @param[in]    strCurrentDir    Current directory.
     * @return Whether the function call succeed. NC_TRUE indicates success,
     * otherwise indicates failure.
     * @remarks After this function, the path is absolute.
     */
    NC_BOOL Absolute( const NCString &strCurrentDir );
    // static NCString ToNative(const NCString& strPathName);

    // NCString MakePath(const NCString& strPathName);

    /**
     * @brief Check whether the path is valid
     *
     * @return NC_TRUE indicates that the path is valid, otherwise the path is
     * invalid.
     */
    NC_BOOL isValid() const;

    /**
     * @brief Check whether a path is valid
     *
     * @param[in]    strPathName    The path to be checked.
     * @return NC_TRUE indicates that the path is valid, otherwise the path is
     * invalid.
     */
    static NC_BOOL isValid( const NCString &strPathName );

    /**
     * @brief Check whether the path is a relative path
     *
     * @return NC_TRUE indicates that the path is a relative path, otherwise the
     * path is not a relative path.
     */
    NC_BOOL isRelative() const;

    /**
     * @brief Check whether a path is a relative path
     *
     * @param[in]    strPathName    The path to be checked.
     * @return NC_TRUE indicates that the path is a relative path, otherwise the
     * path is not a relative path.
     */
    static NC_BOOL isRelative( const NCString &strPathName );

    /**
     * @brief Check whether the path is a absolute path
     *
     * @return NC_TRUE indicates that the path is a absolute path, otherwise the
     * path is not a absolute path.
     */
    NC_BOOL isAbsolute() const;

    /**
     * @brief Check whether a path is a absolute path
     *
     * @param[in]    strPathName    The path to be checked.
     * @return NC_TRUE indicates that the path is a absolute path, otherwise the
     * path is not a absolute path.
     */
    static NC_BOOL isAbsolute( const NCString &strPathName );

    /**
     * @brief Check whether the path is a root path
     *
     * @return NC_TRUE indicates that the path is a root path, otherwise the path
     * is not a root path.
     */
    NC_BOOL isRoot() const;

    /**
     * @brief Check whether a path is a root path
     *
     * @param[in]    strPathName    The path to be checked.
     * @return NC_TRUE indicates that the path is a root path, otherwise the path
     * is not a root path.
     */
    static NC_BOOL isRoot( const NCString &strPathName );

    /**
     * @brief Get the drive name of this path
     */
    NCString DriveName() const;

    /**
     * @brief Get upper level directory of this path
     */
    NCString UpperDir() const;

    // *
    // * @brief Get the absolute upper level directory of this path
    // */
    // NCString AbsoluteUpperDir() const;

    /**
     * @brief Get the name of the path
     */
    NCString PathName() const;

    // NCString DirName() const;

    /**
     * @brief Get the file name of the path
     */
    NCString FileName() const;

    /**
     * @brief Get the base name of the file name
     */
    NCString BaseName() const;

    /**
     * @brief Get the full base name of the file name
     */
    NCString FullBaseName() const;

    /**
     * @brief Get the suffix of the file name
     */
    NCString Suffix() const;

    /**
     * @brief Get the full suffix of the file name
     */
    NCString FullSuffix() const;

    /**
     * @brief Append a sub file name to this path
     */
    NC_BOOL append( const NCString &strFileName );

   private:
    NCPathImpl *m_ptr;
    NCPath( const NCPath & );
    NCPath &operator=( const NCPath & );
};
OSAL_END_NAMESPACE
#endif /* NCPATH_H */
/* EOF */
