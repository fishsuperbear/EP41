#include "util/codec_util.h"

#include <Poco/Base64Decoder.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace advc {

unsigned char CodecUtil::ToHex(const unsigned char &x) {
    return x > 9 ? (x - 10 + 'A') : x + '0';
}

std::string CodecUtil::EncodeKey(const std::string &key) {
    std::string encodedKey = "";
    std::size_t length = key.length();
    for (size_t i = 0; i < length; ++i) {
        if (isalnum((unsigned char)key[i]) || (key[i] == '-') || (key[i] == '_') ||
            (key[i] == '.') || (key[i] == '~') || (key[i] == '/')) {
            encodedKey += key[i];
        } else {
            encodedKey += '%';
            encodedKey += ToHex((unsigned char)key[i] >> 4);
            encodedKey += ToHex((unsigned char)key[i] % 16);
        }
    }
    return encodedKey;
}

std::string CodecUtil::UrlEncode(const std::string &str) {
    std::string encodedUrl = "";
    std::size_t length = str.length();
    for (size_t i = 0; i < length; ++i) {
        if (isalnum((unsigned char)str[i]) || (str[i] == '-') || (str[i] == '_') ||
            (str[i] == '.') || (str[i] == '~')) {
            encodedUrl += str[i];
        } else {
            encodedUrl += '%';
            encodedUrl += ToHex((unsigned char)str[i] >> 4);
            encodedUrl += ToHex((unsigned char)str[i] % 16);
        }
    }
    return encodedUrl;
}
std::string CodecUtil::Base64Decode(const std::string &plain_text) {
    std::istringstream istr(plain_text);
    std::ostringstream ostr;
    Poco::Base64Decoder b64in(istr);
    copy(std::istreambuf_iterator<char>(b64in),
         std::istreambuf_iterator<char>(),
         std::ostreambuf_iterator<char>(ostr));
    return ostr.str();
}

}  // namespace advc
