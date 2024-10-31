#ifndef AES_CBC_H_
#define AES_CBC_H_

#include <openssl/aes.h>

#include <string>
#include <vector>

namespace advc {

class AesCbcEncryptor {
   public:
    static std::string AesCbcEncryptString(const std::string &in_data,
                                           const std::string &key_base64,
                                           int key_size);
    static std::string AesCbcDecryptString(const std::string &in_data,
                                           const std::string &key_base64,
                                           int key_size);
    // step_size=16*n
    static bool AesCbcEncryptFile(const std::string &in_file,
                                  const int step_size,
                                  const std::string &key_base64,
                                  const int key_size,
                                  const std::string &out_file);
    // step_size=16*n
    static bool AesCbcDecryptFile(const std::string &in_file,
                                  const int step_size,
                                  const std::string &key_base64,
                                  const int key_size,
                                  const std::string &out_file);

   private:
    static bool EnPaddingPkcs5(unsigned char *in_data, int &data_len);
    static bool DepaddingPkcs5(unsigned char *in_data, int &data_len);

};  // AesEncryptor

}  // namespace advc

#endif  // AES_CBC_H_
