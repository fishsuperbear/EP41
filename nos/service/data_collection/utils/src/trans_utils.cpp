/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: time_utils.cpp
 * @Date: 2023/11/06
 * @Author: kun
 * @Desc: --
 */

#include <chrono>
#include <sstream>
#include <iomanip>

#include "utils/include/trans_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

std::map<char, std::function<std::string(void)>> TransUtils::getTimeMap() {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::ostringstream nowTimeSS;
    nowTimeSS << std::put_time(std::localtime(&nowTime), "%y%Y%m%d%H%I%M%S");
    std::string nowTimeStr = nowTimeSS.str();
    std::map<char, std::function<std::string(void)>> timeMap;
    timeMap['y'] = [nowTimeStr]{return nowTimeStr.substr(0, 2);};
    timeMap['Y'] = [nowTimeStr]{return nowTimeStr.substr(2, 4);};
    timeMap['m'] = [nowTimeStr]{return nowTimeStr.substr(6, 2);};
    timeMap['d'] = [nowTimeStr]{return nowTimeStr.substr(8, 2);};
    timeMap['H'] = [nowTimeStr]{return nowTimeStr.substr(10, 2);};
    timeMap['I'] = [nowTimeStr]{return nowTimeStr.substr(12, 2);};
    timeMap['M'] = [nowTimeStr]{return nowTimeStr.substr(14, 2);};
    timeMap['S'] = [nowTimeStr]{return nowTimeStr.substr(16, 2);};
    return timeMap;
}

std::string TransUtils::stringTransFileName(const std::string &inputStr) {
    std::map<char, std::function<std::string(void)>> timeMap = getTimeMap();
    std::string fileNameStr;
    for (uint i = 0; i < inputStr.size(); i++) {
        if (inputStr[i] == '%') {
            if ((i + 1) < inputStr.size()) {
                if (timeMap.count(inputStr[i + 1]) == 1) {
                    fileNameStr = fileNameStr + timeMap[inputStr[i + 1]]();
                    i++;
                    continue;
                }
            }
        }
        fileNameStr = fileNameStr + inputStr[i];
    }
    return fileNameStr;
}

const std::string TransUtils::TRIGGERID_TAG{{"%triggerId"}};

std::string TransUtils::stringTransFileName(const std::string &inputStr, const std::string& triggerId) {
    std::string processedStr;
    if (auto pos = inputStr.find(TRIGGERID_TAG); pos != std::string::npos) {
        processedStr = inputStr.substr(0, pos);
        processedStr += triggerId;
        processedStr += inputStr.substr(pos + TRIGGERID_TAG.size());
    } else {
        return stringTransFileName(inputStr);
    }
    return stringTransFileName(processedStr);
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
