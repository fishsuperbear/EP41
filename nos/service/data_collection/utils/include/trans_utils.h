/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: trans_utils.h
 * @Date: 2023/11/06
 * @Author: kun
 * @Desc: --
 */

#ifndef SERVICE_DATA_COLLECTION_COMMON_UTILS_TRANS_UTILS_H
#define SERVICE_DATA_COLLECTION_COMMON_UTILS_TRANS_UTILS_H

#include <string>
#include <map>
#include <functional>

namespace hozon {
namespace netaos {
namespace dc {

class TransUtils {
public:
    static std::map<char, std::function<std::string(void)>> getTimeMap();
    static std::string stringTransFileName(const std::string &inputStr);
    const static std::string TRIGGERID_TAG;
    static std::string stringTransFileName(const std::string &inputStr, const std::string& triggerId);
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // SERVICE_DATA_COLLECTION_COMMON_UTILS_TRANS_UTILS_H