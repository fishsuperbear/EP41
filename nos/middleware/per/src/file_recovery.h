/*
 * @Author: Z04975 liguoqiang@hozonauto.com
 * @Date: 2023-07-04 17:49:31
 * @LastEditors: Z04975 liguoqiang@hozonauto.com
 * @LastEditTime: 2023-12-16 14:07:29
 * @FilePath: /nos/middleware/per/src/file_recovery.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 文件恢复
 * Created on: Feb 7, 2023
 * Author: liguoqiang
 *
 */
#ifndef MIDDLEWARE_PER_SRC_FILE_RECOVERY_H_
#define MIDDLEWARE_PER_SRC_FILE_RECOVERY_H_
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "core/result.h"
#include "core/span.h"
#include "include/kvs_type.h"
#include "include/per_base_type.h"
#include "include/per_error_domain.h"
#include "include/per_utils.h"
namespace hozon {
namespace netaos {
namespace per {
class FileRecovery {
 public:
    bool RecoverHandle(const std::string& filepath, const StorageConfig config);
    bool BackUpHandle(const std::string& filepath, const StorageConfig config);
    bool ResetHandle(const std::string& filepath, const StorageConfig config);
    int DeleteFile(const std::string& filepath, const StorageConfig config);
    bool FileExist(const std::string& filepath);
    bool CheckCrc32(const std::string& filepath);

 private:
    bool readsp(const std::string& filepath, uint32_t& sp);
    bool readsp(const std::string& filepath, std::string& sp);
    bool writesp(const std::string& filepath, uint32_t sp);
    void ClearFiles(const std::string& filepath, const StorageConfig config);
    void GetFiles(std::string dirPath, std::string nstr, std::deque<std::string>& files);
    template <class T>
    T stringToNum(const std::string& str);
    template <class T>
    std::string NumToString(const T t);
    uint32_t crc32(uint8_t* buf, int len);
    bool CopybakToOriginFile(std::string src, std::string dest);
    // void GenerateCRC32_Table();
    // void init_CRC32_table();
    // unsigned int GetCRC32(unsigned char* buf, unsigned int len);
    static bool Mkdir(std::string filepath, bool isfile);
};

}  // namespace per
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_PER_SRC_FILE_RECOVERY_H_
