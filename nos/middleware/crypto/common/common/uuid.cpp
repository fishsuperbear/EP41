#include "common/uuid.h"

#include <stdio.h>
#include <string.h>
#include <vector>
#include <iomanip>

#include "common/crypto_error_domain.h"

namespace hozon {
namespace netaos {
namespace crypto {
const size_t DEFAULT_UUID_STR_LEN = 36;
const char* DEFAULT_UUID_STR = "00000000-0000-0000-0000-000000000000";
const char UUID_DEL = '-';
const size_t UUID_1ST_DEL = 8;
const size_t UUID_2ND_DEL = 13;
const size_t UUID_3RD_DEL = 18;
const size_t UUID_4TH_DEL = 23;

std::string Uuid::ToUuidStr() const noexcept {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << ((mQwordMs & 0xFFFFFFFF00000000ULL) >> 32);
    oss << "-";
    oss << std::hex << std::setfill('0') << std::setw(4) << ((mQwordMs & 0x00000000FFFF0000ULL) >> 16);
    oss << "-";
    oss << std::hex << std::setfill('0') << std::setw(4) << ((mQwordMs & 0x000000000000FFFFULL));
    oss << "-";
    oss << std::hex << std::setfill('0') << std::setw(4) << ((mQwordLs & 0xFFFF000000000000ULL) >> 48);
    oss << "-";
    oss << std::hex << std::setfill('0') << std::setw(12) << ((mQwordLs & 0x0000FFFFFFFFFFFFULL));
    return oss.str();

}

bool CheckUuidStr(std::string uuid_str) {
    // Check length
    if (uuid_str.size() != std::string(DEFAULT_UUID_STR).size()) {
        return false;
    }

    // Check format;
    if ((uuid_str.find(UUID_DEL, 0) != UUID_1ST_DEL)
        || (uuid_str.find(UUID_DEL, UUID_1ST_DEL + 1) != UUID_2ND_DEL)
        || (uuid_str.find(UUID_DEL, UUID_2ND_DEL + 1) != UUID_3RD_DEL)
        || (uuid_str.find(UUID_DEL, UUID_3RD_DEL + 1) != UUID_4TH_DEL)) {
        return false;
    }

    return true;
}

netaos::core::Result<Uuid> FromString(std::string uuid_str) noexcept {
    
    if (!CheckUuidStr(uuid_str)) {
        return netaos::core::Result<Uuid>::FromError(CryptoErrc::kCommonErr);
    }

    uint64_t high = 0;
    uint64_t low = 0;
    for (size_t i = 0; i < uuid_str.size(); ++i) {
        if ((i < 19) && (uuid_str[i] != UUID_DEL)) {
            if (('0' <= uuid_str[i]) && (uuid_str[i] <= '9')) {
                high = high << 4;
                high |= uuid_str[i] - '0';
            }
            else if ('a' <= uuid_str[i] && uuid_str[i] <= 'f') {
                high = high << 4;
                high |= uuid_str[i] - 'a' + 10;
            }
            else if ('-' == uuid_str[i]) {
                continue;
            }
            else {
                return netaos::core::Result<Uuid>::FromError(CryptoErrc::kCommonErr);
            }
        }
        else {
            if (('0' <= uuid_str[i]) && (uuid_str[i] <= '9')) {
                low = low << 4;
                low |= uuid_str[i] - '0';
            }
            else if ('a' <= uuid_str[i] && uuid_str[i] <= 'f') {
                low = low << 4;
                low |= uuid_str[i] - 'a' + 10;
            }
            else if ('-' == uuid_str[i]) {
                continue;
            }
            else {
                return netaos::core::Result<Uuid>::FromError(CryptoErrc::kCommonErr);
            }
        }
    }

    return Uuid {low, high};
}

void GetCmdOutput(const std::string &cmd, std::vector<char>& buf) {
    std::unique_ptr<FILE, decltype(&pclose)> file(popen(cmd.c_str(), "r"), pclose);
    // FILE *fp=nullptr;
    // if((fp=popen(cmd.c_str(),"r"))==nullptr)
    // {
    //     return;
    // }
    char read_str[10 * 1024];
    while(int size = fread(read_str, 1, sizeof(read_str), file.get()))
    {
        buf.resize(buf.size() + size);
        ::memcpy(buf.data() + buf.size() - size, read_str, size);
    }
    // pclose(fp);
}

netaos::core::Result<Uuid> MakeVersion4Uuid() noexcept {
    
    std::vector<char> output;
    std::string cmd = "cat /proc/sys/kernel/random/uuid";
    GetCmdOutput(cmd, output);

    if (output.size() <= 0) {
        return netaos::core::Result<Uuid>::FromValue({0, 0});
    }
    // covert to char \0 push in vector，size add 1
    std::string uuid_str(output.data(), output.size() - 1);
    if (!CheckUuidStr(uuid_str)) {
        return netaos::core::Result<Uuid>::FromValue({0, 0});
    }

    auto uuid_res = FromString(uuid_str);
    return uuid_res;
}

}
}
}