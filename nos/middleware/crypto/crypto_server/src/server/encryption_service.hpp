#ifndef ARA_CRYPTO_KEYS_ENCRYPTION_SERVICE_H_
#define ARA_CRYPTO_KEYS_ENCRYPTION_SERVICE_H_
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/crypto.h>
#include <fstream>

#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {

class EncryptionService {
public:

static EncryptionService& Instance() {
    static EncryptionService instance;
    return instance;
}

int do_FileCrypt_ReturnString(const std::string& inPath, std::string& outString, int do_encrypt)
{
    if (inPath.empty()) {
        CRYP_ERROR << "do_FileCrypt, inPath is empty :"<< inPath;;
        return 0;
    }
    FILE* in = fopen(inPath.c_str(), "rb");
    if (!in) {
        CRYP_ERROR <<"do_FileCrypt, can not find file";
        return 0;
    }

    std::stringstream out;
    unsigned char inbuf[1024], outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    int inlen, outlen;
    EVP_CIPHER_CTX* ctx;

    unsigned char key[] = "0123456789abcdeF";
    unsigned char iv[] = "1234567887654321";

    ctx = EVP_CIPHER_CTX_new();
    if (!EVP_CipherInit_ex2(ctx, EVP_aes_128_cbc(), NULL, NULL, do_encrypt, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    OPENSSL_assert(EVP_CIPHER_CTX_get_key_length(ctx) == 16);
    OPENSSL_assert(EVP_CIPHER_CTX_get_iv_length(ctx) == 16);

    if (!EVP_CipherInit_ex2(ctx, NULL, key, iv, do_encrypt, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }

    while (1) {
        inlen = fread(inbuf, 1, 1024, in);
        if (inlen <= 0) {
            CRYP_INFO << "inlen:" << std::dec << inlen;
            break;
        }
        CRYP_INFO << "inlen:" << inlen;
        if (!EVP_CipherUpdate(ctx, outbuf, &outlen, inbuf, inlen)) {
            EVP_CIPHER_CTX_free(ctx);
            return 0;
        }
        out.write(reinterpret_cast<const char*>(outbuf), outlen);
    }
    if (!EVP_CipherFinal_ex(ctx, outbuf, &outlen)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    out.write(reinterpret_cast<const char*>(outbuf), outlen);
    fclose(in);
    CRYP_INFO << "encrytp finish.";
    EVP_CIPHER_CTX_free(ctx);
    outString = out.str();
    return 1;
}

int do_FileCrypt(const std::string& inPath, const std::string& outPath, int do_encrypt)
{
    if (inPath.empty()) {
        CRYP_ERROR <<"do_FileCrypt, inPath is empty :" << inPath;
        return 0;
    }
    FILE* in = fopen(inPath.c_str(), "rb");
    FILE* out = fopen(outPath.c_str(), "wb");
    if (!in || !out) {
        CRYP_ERROR <<"do_FileCrypt, can not find or create file";
        return 0;
    }
    /* Allow enough space in output buffer for additional block */
    unsigned char inbuf[1024],outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    // unsigned char inbuf[] = "0123456789abcdeF";
    int inlen, outlen;
    EVP_CIPHER_CTX *ctx;

    unsigned char key[] = "0123456789abcdeF";
    unsigned char iv[] = "1234567887654321";

    /* Don't set key or IV right away; we want to check lengths */
    ctx = EVP_CIPHER_CTX_new();
    if (!EVP_CipherInit_ex2(ctx, EVP_aes_128_cbc(), NULL, NULL,do_encrypt, NULL)) {

        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    OPENSSL_assert(EVP_CIPHER_CTX_get_key_length(ctx) == 16);
    OPENSSL_assert(EVP_CIPHER_CTX_get_iv_length(ctx) == 16);

    if (!EVP_CipherInit_ex2(ctx, NULL, key, iv, do_encrypt, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }

   while(1) {
        inlen = fread(inbuf, 1, 1024, in);
        if (inlen <= 0){
            CRYP_INFO << "inlen:"<<std::dec<<inlen;
            break;
        }
        CRYP_INFO << "inlen:"<<inlen;
        if (!EVP_CipherUpdate(ctx, outbuf, &outlen, inbuf, inlen)) {
            EVP_CIPHER_CTX_free(ctx);
            return 0;
        }
        fwrite(outbuf, 1, outlen, out);
    }
    if (!EVP_CipherFinal_ex(ctx, outbuf, &outlen)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    fwrite(outbuf, 1, outlen, out);
    fclose(in);
    fclose(out);
    CRYP_INFO << "encrytp finish.";
    EVP_CIPHER_CTX_free(ctx);
    return 1;
};

private:
    EncryptionService() {}
    EncryptionService(const EncryptionService&) = delete;
    EncryptionService& operator=(const EncryptionService&) = delete;
};

}  // namespace keys
}  // namespace hozon
}  // namespace neta
#endif  // #define ARA_CRYPTO_KEYS_IMP_JSON_PARSER_H_