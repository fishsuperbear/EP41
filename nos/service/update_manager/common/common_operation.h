/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: data definition
 */
#ifndef OTA_COMM_OPRATION_DEF_H
#define OTA_COMM_OPRATION_DEF_H

#include <stdint.h>
#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <thread>

namespace hozon {
namespace netaos {
namespace update {

bool PathExists(const std::string &pathName);
bool PathClear(const std::string &pathName);
bool PathRemove(const std::string &pathName);
bool PathCreate(const std::string &pathName);
bool FileRecovery(const std::string &src, const std::string &dst);
bool UnzipFile(const std::string &zipFileName, const std::string &unzipPath);
bool FileMount(const std::string &pathName, const std::string &mountPath);
bool FileUmount(const std::string &umountPath);
bool SystemSync();
bool GetAbsolutePath(const std::string& relativePath, std::string& absolutePath);

std::string getFileName(const std::string &FilePath);
int32_t getFileSize(const std::string &FilePath);
int16_t createFile(const std::string& filePath);
int16_t writeToFile(const std::string& filePath, const std::string& content);
int16_t readFile(const std::string& filePath, std::string& content);

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // OTA_COMM_OPRATION_DEF_H
