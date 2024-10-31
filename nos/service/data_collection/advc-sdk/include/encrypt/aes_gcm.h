#ifndef AES_GCM_H_
#define AES_GCM_H_
#include <string>

namespace advc {
class AesGcmEncryptor {
   public:
    static std::string AesGcmEncryptString(const std::string &gcm_plaintext,
                                           const std::string &gcm_key_base64,
                                           int gcm_key_size);
    static std::string AesGcmDecryptString(const std::string &gcm_ciphertext,
                                           const std::string &gcm_key_base64,
                                           int gcm_key_size);
    // step_size=16*n
    static bool AesGcmEncryptFile(const std::string &gcm_plaintext_file,
                                  const int step_size,
                                  const std::string &gcm_key_base64,
                                  int gcm_key_size,
                                  const std::string &gcm_ciphertext_file);
    // step_size=16*n
    static bool AesGcmDecryptFile(const std::string &gcm_ciphertext_file,
                                  const int step_size,
                                  const std::string &gcm_key_base64,
                                  int gcm_key_size,
                                  const std::string &gcm_plaintext_file);
};
}  // namespace advc
#endif
