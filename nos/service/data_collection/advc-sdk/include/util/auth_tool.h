#ifndef UTIL_AUTHTOOl_H
#define UTIL_AUTHTOOl_H

#include <cstdint>

#include <map>
#include <string>
#include <vector>
#include "time_util.h"
#include "util/noncopyable.h"

namespace advc {

    class AuthTool : private NonCopyable {
    public:

        static std::string getV4Date();

        static std::string getV4Time();

        static std::string ToSha256Hex(const std::string &in);

        static std::string StringToHex(const std::string &data);

        static std::string ContentSha256(const std::string& secret, const std::string &body);

        static std::string ContentSha256Hex(std::string secret, const std::string &body);
    };

}  // namespace advc

#endif
