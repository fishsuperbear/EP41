#include "encrypt/aes_cbc.h"

#include <openssl/aes.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <iostream>

#include "advc_sys_config.h"
#include "util/codec_util.h"

namespace advc {

std::string AesCbcEncryptor::AesCbcEncryptString(const std::string &in_data,
                                                 const std::string &key_base64,
                                                 int key_size) {
    if (key_base64.empty() || in_data.empty()) {
        return "";
    }
    AES_KEY aes_key;
    srand((int)time(NULL));
    std::string key = CodecUtil::Base64Decode(key_base64);
    if (key.length() * 8 != key_size) {
        std::cout << key.length() << " " << key_size << std::endl;
        return "";
    }

    int brc = AES_set_encrypt_key((const unsigned char *)key.data(), key_size,
                                  &aes_key);
    if (brc < 0) {
        return "";
    }
    std::size_t data_len = in_data.length();
    // auto padding, add 0 at tail
    // iv + output
    std::size_t out_len =
        AES_BLOCK_SIZE + ((data_len % AES_BLOCK_SIZE == 0) ? data_len : (data_len / AES_BLOCK_SIZE + 1) * AES_BLOCK_SIZE);
    unsigned char *out = (unsigned char *)malloc(out_len);
    memset(out, 0, out_len);
    unsigned char iv[AES_BLOCK_SIZE];
    for (int i = 0; i < AES_BLOCK_SIZE; ++i) {
        iv[i] = rand() % 256;
        out[i] = iv[i];
    }
    AES_cbc_encrypt((const unsigned char *)in_data.data(),
                    out + AES_BLOCK_SIZE, data_len, &aes_key, iv, AES_ENCRYPT);

    std::string out_data = std::string((const char *)out, out_len);
    free(out);
    return out_data;
}

std::string AesCbcEncryptor::AesCbcDecryptString(const std::string &in_data,
                                                 const std::string &key_base64,
                                                 int key_size) {
    if (key_base64.empty() || in_data.empty()) {
        return "";
    }
    AES_KEY aes_key;
    std::string key = CodecUtil::Base64Decode(key_base64);
    if (key.length() * 8 != key_size) {
        return "";
    }
    unsigned char iv[AES_BLOCK_SIZE];
    memcpy(iv, in_data.data(), AES_BLOCK_SIZE);
    int brc = AES_set_decrypt_key((const unsigned char *)key.data(), key_size,
                                  &aes_key);
    if (brc < 0) {
        return "";
    }
    std::size_t data_len = in_data.length() - AES_BLOCK_SIZE;
    unsigned char *out = (unsigned char *)malloc(data_len);
    memset(out, 0, data_len);
    AES_cbc_encrypt((const unsigned char *)in_data.data() + AES_BLOCK_SIZE,
                    out, data_len, &aes_key, iv, AES_DECRYPT);
    // rm auto padding
    while (data_len > 0 && out[data_len - 1] == 0) {
        --data_len;
    }
    std::string out_data = std::string((const char *)out, data_len);
    free(out);
    return out_data;
}

// step_size must be 16*n
bool AesCbcEncryptor::AesCbcEncryptFile(const std::string &in_file,
                                        const int step_size,
                                        const std::string &key_base64,
                                        const int key_size,
                                        const std::string &out_file) {
    if (key_base64.empty()) {
        return false;
    }
    AES_KEY aes_key;
    std::string key = CodecUtil::Base64Decode(key_base64);
    if (key.length() * 8 != key_size) {
        return false;
    }

    int brc = AES_set_encrypt_key((const unsigned char *)key.data(), key_size,
                                  &aes_key);
    if (brc < 0) {
        return false;
    }

    std::ifstream in_file_io(in_file, std::ios::in | std::ios::binary);
    if (!in_file_io.good()) {
        return false;
    }
    unsigned char *out = (unsigned char *)malloc(step_size);
    unsigned char *step_block = (unsigned char *)malloc(step_size);
    memset(out, 0, step_size);
    memset(step_block, 0, step_size);
    unsigned char iv[AES_BLOCK_SIZE];
    for (int i = 0; i < AES_BLOCK_SIZE; ++i) {
        iv[i] = rand() % 256;
    }

    int in_len;
    std::ofstream out_file_io(out_file, std::ios::out | std::ios::binary);
    out_file_io.write((char *)iv, AES_BLOCK_SIZE);
    int count = 0;
    while (!in_file_io.eof()) {
        in_file_io.read((char *)step_block, step_size);
        in_len = in_file_io.gcount();
        // padding for file tail, avoid openssl auto padding
        if (in_file_io.eof()) {
            if (in_len < step_size) {
                EnPaddingPkcs5(step_block, in_len);
            } else {
                AES_cbc_encrypt((const unsigned char *)step_block, out, in_len,
                                &aes_key, iv, AES_ENCRYPT);
                out_file_io.write((const char *)out, in_len);
                in_len = 0;
                EnPaddingPkcs5(step_block, in_len);
            }
        }
        AES_cbc_encrypt((const unsigned char *)step_block, out, in_len,
                        &aes_key, iv, AES_ENCRYPT);
        out_file_io.write((const char *)out, in_len);
        count++;
        if (count % AdvcSysConfig::GetEncryptSleepCount() == 0) {
            usleep(AdvcSysConfig::GetEncryptSleepTimeInMicrosecond());
        }
    }

    in_file_io.close();
    out_file_io.close();
    free(out);
    free(step_block);
    return true;
}

bool AesCbcEncryptor::AesCbcDecryptFile(const std::string &in_file,
                                        const int step_size,
                                        const std::string &key_base64,
                                        const int key_size,
                                        const std::string &out_file) {
    if (key_base64.empty()) {
        return false;
    }
    AES_KEY aes_key;
    std::string key = CodecUtil::Base64Decode(key_base64);
    if (key.length() * 8 != key_size) {
        return false;
    }

    int brc = AES_set_decrypt_key((const unsigned char *)key.data(), key_size,
                                  &aes_key);
    if (brc < 0) {
        return false;
    }

    std::ifstream in_file_io(in_file, std::ios::in | std::ios::binary);
    if (!in_file_io.good()) {
        return false;
    }
    unsigned char *out = (unsigned char *)malloc(step_size);
    unsigned char *step_block = (unsigned char *)malloc(step_size);
    memset(out, 0, step_size);
    memset(step_block, 0, step_size);
    unsigned char iv[AES_BLOCK_SIZE];
    in_file_io.read((char *)iv, AES_BLOCK_SIZE);

    int in_len;
    std::ofstream out_file_io(out_file, std::ios::out | std::ios::binary);
    int count = 0;
    while (!in_file_io.eof()) {
        in_file_io.read((char *)step_block, step_size);
        in_len = in_file_io.gcount();
        AES_cbc_encrypt((const unsigned char *)step_block, out, in_len,
                        &aes_key, iv, AES_DECRYPT);
        if (in_file_io.eof()) {
            DepaddingPkcs5(out, in_len);
        }
        out_file_io.write((const char *)out, in_len);
        count++;
        if (count % AdvcSysConfig::GetDecryptSleepCount() == 0) {
            usleep(AdvcSysConfig::GetDecryptSleepTimeInMicrosecond());
        }
    }
    in_file_io.close();
    out_file_io.close();
    free(out);
    free(step_block);
    return true;
}

bool AesCbcEncryptor::EnPaddingPkcs5(unsigned char *in_data, int &data_len) {
    unsigned char pad_len = AES_BLOCK_SIZE - data_len % AES_BLOCK_SIZE;
    for (int i = 0; i < pad_len; ++i) {
        in_data[data_len] = pad_len;
        ++data_len;
    }
    return true;
}
bool AesCbcEncryptor::DepaddingPkcs5(unsigned char *in_data, int &data_len) {
    unsigned char tail = in_data[data_len - 1];
    data_len -= tail;
    return true;
}

}  // namespace advc