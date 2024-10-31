#ifndef CODEC_UTIL_H
#define CODEC_UTIL_H

#include <cstdint>

#include <string>

namespace advc {

    class CodecUtil {
    public:
        /**
         * @brief 将字符x转成十六进制 (x的值[0, 15])
         *
         * @param x
         *
         * @return 十六进制字符
         */
        static unsigned char ToHex(const unsigned char &x);


        static std::string EncodeKey(const std::string &key);

        /**
         * @brief 对字符串进行URL编码
         *
         * @param str   带编码的字符串
         *
         * @return  经过URL编码的字符串
         */
        static std::string UrlEncode(const std::string &str);

        /**
         * @brief 对字符串进行base64解码
         *
         * @param plainText  待解码的字符串
         *
         * @return 解码后的字符串
         */
        static std::string Base64Decode(const std::string &plainText);


    };

}  // namespace advc
#endif
