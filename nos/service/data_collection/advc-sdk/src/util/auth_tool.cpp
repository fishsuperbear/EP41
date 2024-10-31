
#include "util/auth_tool.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "util/advc_hmac.h"
#include "util/string_util.h"
#include "util/time_util.h"

namespace advc {

std::string AuthTool::getV4Date() {
    time_t time_utc = Time::time_util->getLocalTime();
    struct tm tm_gmt {};
    gmtime_r(&time_utc, &tm_gmt);
    char cur_time[256];
    strftime(cur_time, 256, "%Y%m%d", &tm_gmt);
    return cur_time;
}

std::string AuthTool::getV4Time() {
    time_t time_utc = Time::time_util->getLocalTime();
    struct tm tm_gmt {};
    gmtime_r(&time_utc, &tm_gmt);
    char cur_time[256];
    strftime(cur_time, 256, "%Y%m%dT%H%M%SZ", &tm_gmt);
    return cur_time;
}

std::string AuthTool::StringToHex(const std::string &data) {
    const std::string hex = "0123456789abcdef";

    std::stringstream ss;

    for (std::string::size_type i = 0; i < data.size(); ++i)
        ss << hex[(unsigned char)data[i] >> 4] << hex[(unsigned char)data[i] & 0xf];
    return ss.str();
}

std::string AuthTool::ToSha256Hex(const std::string &in) {
    std::string ret = sha256(in);
    return ret;
}

std::string AuthTool::ContentSha256(const std::string &key, const std::string &msg) {
    unsigned char *mac = nullptr;
    unsigned int mac_length = 0;

    HmacEncode("sha256", key, key.length(), msg.c_str(), msg.length(), mac, mac_length);
    std::string out;
    out.assign((char *)mac, mac_length);
    if (mac) {
        free(mac);
    }
    return out;
}

std::string AuthTool::ContentSha256Hex(std::string key, const std::string &msg) {
    unsigned char *mac = nullptr;
    unsigned int mac_length = 0;

    HmacEncode("sha256", key, key.length(), msg.c_str(), msg.length(), mac, mac_length);
    std::string out;
    out.assign((char *)mac, mac_length);
    std::string ret = StringToHex(out);

    if (mac) {
        free(mac);
    }
    return ret;
}

}  // namespace advc
