#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <unzipper.h>

namespace hozon {
namespace netaos {
namespace logserver {

bool PathExists(const std::string &pathName);
bool PathClear(const std::string &pathName);
bool PathRemove(const std::string &pathName);
bool PathCreate(const std::string &pathName);
bool FileRecovery(const std::string &src, const std::string &dst);
bool UnzipFile(const std::string &zipFileName, const std::string &unzipPath);
std::string getFileName(const std::string &FilePath);
std::uint32_t getFileSize(const std::string &FilePath);

}  // namespace logserver
}  // namespace netaos
}  // namespace hozon

