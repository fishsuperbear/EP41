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
 * @file NCRuntime.h
 * @brief
 * @date 2020-06-01
 *
 */

#ifndef INCLUDE_NCORE_NCRUNTIME_H_
#define INCLUDE_NCORE_NCRUNTIME_H_

#include "osal/ncore/NCAutoSync.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCTypesDefine.h"

OSAL_BEGIN_NAMESPACE

const NCString NC_PLATFORM_CLUSTER( "platformcluster" );

class NCRuntime {
   public:
    /**
     * @brief Get the data path of the specified APP
     * @param cluster Cluster Name(NC_PLATFORM_CLUSTER or others)
     * @param process Process Name
     * @return NCString the data path of the specified APP(eg:/data/data/[clusterName]/[processName]/)
     */
    static NCString getAppDataDir( const NCString& cluster, const NCString& process );

    /**
     * @brief Get the data path of self
     * @return NCString the data path(eg:/data/data/[clusterName]/[processName]/)
     */
    static NCString getAppDataDir();

    /**
     * @brief Get the config path of the specified APP
     * @param cluster Cluster Name(NC_PLATFORM_CLUSTER or others)
     * @param process Process Name
     * @return NCString the config path of the specified APP(eg:/data/apps/[clusterName]/[processName]/)
     */
    static NCString getAppDir( const NCString& cluster, const NCString& process );

    /**
     * @brief Get the config path of self
     * @return NCString the config path(eg:/data/apps/[clusterName]/[processName]/)
     */
    static NCString getAppDir();

    /**
     * @brief Get the cluster path of self
     * @return NCString the cluster path(eg:/data/apps/[clusterName]/)
     */
    static NCString getClusterDir();

    /**
     * @brief Get the temporary directory
     * @return NCString the config path(eg:/tmp/)
     */
    static NCString getTmpDir();

    /**
     * @brief Get the Process'sName
     * @return NCString process'sname
     */
    static NCString getProName();

    /**
     * @brief Get the Cluster'sName
     * @return NCString Cluster'sname
     */
    static NCString getClusterName();

    /**
     * @brief is test mode
     * @return NC_BOOL
     */
    static NC_BOOL isTestMode();

    /**
     * @brief Get the dummy path
     * @return NCString dummy path
     */
    // will delete
    static NCString getTestRootPath();

    /**
     * @brief Get the Machine's Config Dir
     *
     * @return NCString of machine config(eg:/system/etc/ara/)
     */
    static NCString getConfigDir();

    /**
     * @brief Get the intsall path of app
     *
     * @return NCString intsall path of app (eg:/data/apps/)
     */
    static NCString getAppsPath();

    /**
     * @brief Get the data of path
     *
     * @return NCString data of path(eg:/data/data/)
     */
    static NCString getAppsDataPath();

    /**
     * @brief Get the value of Environment
     *
     * @param key the key of environment
     * @return NCString the environment value
     */
    static NCString getEnvValue( const NCString& key );

    /**
     * @brief Set the value of Environment
     *
     * @param key the key of environment
     *        value the value of environment
     *
     * @return NC_BOOL NC_TRUE  set the environment success
     *                 NC_FALSE set the environment failed
     */
    static NC_BOOL setEnvValue( const NCString& key, const NCString& value );

    /**
     * @brief Set the intsall root path for all app
     *
     * @param path the root path for all app
     *
     * @return NC_BOOL NC_TRUE  set the path success
     *                 NC_FALSE set the path failed
     */
    static NC_BOOL setAppsPath( const NCString& path );

    /**
     * @brief Set the Machine's Config Dir
     *
     * @param path the Machine's Config Dir
     *
     * @return NC_BOOL NC_TRUE  set the path success
     *                 NC_FALSE set the path failed
     */
    static NC_BOOL setConfigDir( const NCString& path );
    /**
     * @brief Set the data of path
     *
     * @param path the data of path
     *
     * @return NC_BOOL NC_TRUE  set the path success
     *                 NC_FALSE set the path failed
     */
    static NC_BOOL setAppsDataPath( const NCString& path );
    /**
     * @brief Set the temporary directory
     *
     * @param path the temporary of path
     *
     * @return NC_BOOL NC_TRUE  set the path success
     *                 NC_FALSE set the path failed
     */
    static NC_BOOL setTmpDir( const NCString& path );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCRUNTIME_H_