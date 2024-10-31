
#include "util/advc_hmac.h"

#include <openssl/aes.h>
#include <openssl/crypto.h>
#include <openssl/sha.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>
#include <iostream>
#include <sstream>

#include "openssl/evp.h"
#include "openssl/hmac.h"

using namespace std;

int HmacEncode(const char *algo, const std::string &key, unsigned int key_length, const char *input, unsigned int input_length,
               unsigned char *&output, unsigned int &output_length) {
    const EVP_MD *EVPEngine = NULL;
    int sha512AlgoResult = strcasecmp("sha512", algo);
    int sha256AlgoResult = strcasecmp("sha256", algo);
    int sha1AlgoResult = strcasecmp("sha1", algo);
    int md5AlgoResult = strcasecmp("md5", algo);

    if (sha512AlgoResult == 0) {
        EVPEngine = EVP_sha512();
    } else if (sha256AlgoResult == 0) {
        EVPEngine = EVP_sha256();
    } else if (sha1AlgoResult == 0) {
        EVPEngine = EVP_sha1();
    } else if (md5AlgoResult == 0) {
        EVPEngine = EVP_md5();
    } else {
        cout << algo << " is not supported !" << endl;
        return -1;
    }
    output = (unsigned char *)malloc(EVP_MAX_MD_SIZE);
    memset(output, 0, EVP_MAX_MD_SIZE);
    HMAC_CTX *ctx;
    ctx = HMAC_CTX_new();
    HMAC_Init_ex(ctx, key.c_str(), key_length, EVPEngine, NULL);
    HMAC_Update(ctx, (unsigned char *)input, input_length);
    HMAC_Final(ctx, output, &output_length);
    HMAC_CTX_free(ctx);
    return 0;
}

std::string sha256(const std::string str) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str.c_str(), str.size());
    SHA256_Final(hash, &sha256);
    stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << hex << setw(2) << setfill('0') << (int)hash[i];
    }
    return ss.str();
}
