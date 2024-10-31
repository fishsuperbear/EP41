/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: file_utils.h
 * @Date: 2023/08/15
 * @Author: cheng
 * @Desc: --
 */

#ifndef MIDDLEWARE_TOOLS_COMMON_UTILS_FILE_UTILS_H
#define MIDDLEWARE_TOOLS_COMMON_UTILS_FILE_UTILS_H

#include <string.h>
#include <string>
#include <memory>
#include <vector>

namespace hozon {
namespace netaos {
namespace dc {


class PathUtils {
public:
  // Check whether file exists
  static bool isFileExist(const std::string& file_path);
  // Check whether directory exists
  static bool isDirExist(const std::string& dir_path);
  // Check whether path exists
  static bool isPathExist(const std::string& path);
  // Make sure path.
  static bool createFoldersIfNotExists(const std::string& path);
  // Delete old files if files count is more than max_file in dir.
  static bool removeOldFile(std::string dir_path, uint32_t max_file);
  // Remove file path.
  static bool removeFile(std::string file_path);
  // Remove file path.
  static bool removeFilesInFolder(std::string file_path);
  // Rename file path.
  static bool renameFile(std::string old_file_path, std::string new_file_path);
  // Remove folder path.
  static bool removeFolder(std::string folder_path);
  static bool clearFolder(std::string path);
  // Fast remove folder path.
  static bool fastRemoveFolder(std::string folder_path, std::string fast_name);
  // Get file name from path.
  static std::string getFileName(std::string path);
  // Get folder name from path.
  static std::string getFolderName(std::string path);
  // Read file to buffer
  static std::shared_ptr<std::vector<uint8_t>> readFile(std::string file_path);

  static bool readFile(std::string file_path, std::vector<char>& content);
  // Write buf to file.
  static bool writeFile(std::string file_path, std::shared_ptr<std::vector<uint8_t>> buf);
  static bool writeFile(std::string file_path, std::string& str);
  static bool getFiles(std::string path, std::string filePattern, bool searchSubPath, std::vector<std::string>& files) ;
  static bool getFiles(std::string path, std::vector<std::string>& files, bool searchSubPath = false) ;
  static std::string getFilePath(std::string folder_path, std::string file_name);

  // Get running mode
  static uint8_t runningMode();
  // Set running mode
  static void setRunningMode(uint8_t running_mode);

  static constexpr const char* debugModeFilePath = "/opt/usr/col/bag/dc_debug_mode_on";
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_TOOLS_COMMON_UTILS_FILE_UTILS_H
