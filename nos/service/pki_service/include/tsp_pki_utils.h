/*
 * Copyright (c) hozonauto. 2021-2022. All rights reserved.
 * Description: TspPkiUtils class definition.
 */

#ifndef V2C_TSP_PKI_TSP_PKI_UTILS_H
#define V2C_TSP_PKI_TSP_PKI_UTILS_H

#include <string>


namespace hozon {
namespace netaos {
namespace tsp_pki {

class TspPkiUtils {
   public:
    // Get data time in second unit
    static uint64_t GetDataTimeSec();
    // Get management time in second unit
    static uint64_t GetMgmtTimeSec();
    // Get app directory.
    static std::string GetAppDirectory();
    // Convert time tick to human readable string.
    static std::string ConvertTime2ReadableStr(uint64_t sec);
    // Set thread name for current thread.
    static void SetThreadName(std::string name);
    // Check whether file exists
    static bool IsFileExist(const std::string& file_path);
    // Check whether directory exists
    static bool IsDirExist(const std::string& dir_path);
    // Check whether path exists
    static bool IsPathExist(const std::string& path);
    // Make sure path.
    static bool MakeSurePath(const std::string& path);
    // Normalize path (linux style).
    static std::string NormalizePath(const std::string& path);
    static int64_t GetFileSize(const std::string& file);
};

}  // namespace tsp_pki
}
}  // namespace hozon
#endif