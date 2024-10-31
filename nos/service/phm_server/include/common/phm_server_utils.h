
#ifndef PHM_UTILS_H
#define PHM_UTILS_H

#include <mutex>
#include <memory>
#include <vector>
#include "phm_server/include/common/phm_server_def.h"

namespace hozon {
namespace netaos {
namespace phm_server {

class PHMUtils {

   public:
    // Get time in micro second unit
    static uint64_t GetTimeMicroSec();
    // Get time in second unit
    static uint32_t GetTimeSec();
    // Get data time macro second in macro second unit
    static uint32_t GetTimeMicroSecInSec();
    // Get time stamp
    static Timestamp GetTimestamp();

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

    // Copy file path.
    static bool CopyFile(const std::string& from_path, const std::string& to_path);
    // Remove file path.
    static bool RemoveFile(std::string file_path);
    // Rename file path.
    static bool RenameFile(const std::string& old_path, const std::string& new_path);
    // Get file name from path.
    static std::string GetFileName(std::string path);
    // Read file to buffer
    static std::shared_ptr<std::vector<uint8_t>> ReadFile(std::string file_path);
    // Write buf to file.
    static bool WriteFile(std::string file_path, std::shared_ptr<std::vector<uint8_t>> buf);
    // Format time into string that can be used in file name.
    static std::string FormatTimeStrForFileName(time_t unix_time);

    static uint64_t GetCurrentTime();
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_UTILS_H