#include "encrypt/aes_gcm.h"

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "advc_sys_config.h"
#include "util/codec_util.h"

namespace advc {

std::string AesGcmEncryptor::AesGcmEncryptString(
    const std::string &gcm_plaintext, const std::string &gcm_key_base64,
    int gcm_key_size) {
    const int tag_size = 16;
    const int gcm_iv_size = 12;
    if (gcm_plaintext.empty() || gcm_key_base64.empty()) {
        return "";
    }
    srand((int)time(NULL));
    std::string key = CodecUtil::Base64Decode(gcm_key_base64);
    if (key.length() * 8 != gcm_key_size) {
        return "";
    }
    if (gcm_plaintext.size() < 1) {
        return "";
    }
    if (256 != gcm_key_size && 128 != gcm_key_size) {
        return "";
    }
    EVP_CIPHER_CTX *ctx;
    unsigned char *outbuf = NULL;
    unsigned char *gcm_tag = NULL;
    unsigned char *gcm_iv = (unsigned char *)malloc(gcm_iv_size);
    outbuf =
        (unsigned char *)malloc(gcm_iv_size + tag_size + gcm_plaintext.size());
    gcm_tag = outbuf + gcm_iv_size;
    for (int i = 0; i < gcm_iv_size; ++i) {
        gcm_iv[i] = rand() % 256;
        outbuf[i] = gcm_iv[i];
    }

    ctx = EVP_CIPHER_CTX_new();
    /* Set cipher type and mode */
    if (gcm_key_size == 256) {
        EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    } else {
        EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    /* Set IV length if default 96 bits is not appropriate */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, gcm_iv_size, NULL);
    /* Initialise key and IV */

    /* Zero or more calls to specify any AAD */
    /* Encrypt plaintext */
    int outlen = 0;
    EVP_EncryptUpdate(ctx, outbuf + gcm_iv_size + tag_size, &outlen,
                      (unsigned char *)gcm_plaintext.data(),
                      gcm_plaintext.size());

    /* Output encrypted block */
    // printf("Ciphertext: outlen:%d\n", *outlen);
    // BIO_dump_fp(stdout, (const char*)(*outbuf), *outlen);
    /* Finalise: note get no output for GCM */
    EVP_EncryptFinal_ex(ctx, outbuf, &outlen);
    // /* Get tag */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, tag_size, (void *)gcm_tag);
    // /* Output tag */
    // printf("Tag: outlen:%d\n", *outlen);
    // BIO_dump_fp(stdout, (const char*)(*outbuf), 16);
    std::string gcm_ciphertext = std::string(
        (char *)outbuf, gcm_plaintext.size() + gcm_iv_size + tag_size);
    EVP_CIPHER_CTX_free(ctx);
    free(outbuf);
    free(gcm_iv);
    return gcm_ciphertext;
}

std::string AesGcmEncryptor::AesGcmDecryptString(
    const std::string &gcm_ciphertext, const std::string &gcm_key_base64,
    int gcm_key_size) {
    const int tag_size = 16;
    const int gcm_iv_size = 12;

    if (gcm_ciphertext.empty() || gcm_key_base64.empty()) {
        return "";
    }
    if (gcm_ciphertext.size() < (gcm_iv_size + tag_size)) {
        return "";
    }
    if (256 != gcm_key_size && 128 != gcm_key_size) {
        return "";
    }
    const unsigned char *gcm_tag =
        (const unsigned char *)gcm_ciphertext.data() + gcm_iv_size;
    unsigned char *gcm_iv = (unsigned char *)malloc(gcm_iv_size);
    for (int i = 0; i < gcm_iv_size; ++i) {
        gcm_iv[i] = (unsigned char)gcm_ciphertext[i];
    }
    std::string key = CodecUtil::Base64Decode(gcm_key_base64);
    if (key.length() * 8 != gcm_key_size) {
        return "";
    }
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    /* Select cipher */
    if (gcm_key_size == 256) {
        EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    } else {
        EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    /* Set IV length, omit for 96 bits */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, gcm_iv_size, NULL);
    /* Specify key and IV */
    //   EVP_DecryptInit_ex(ctx, NULL, NULL, gcm_key, gcm_iv);

    /* Decrypt plaintext */
    unsigned char *outbuf = (unsigned char *)malloc(
        gcm_ciphertext.size() - gcm_iv_size - tag_size + 1);
    int outlen = 0;
    *(outbuf + gcm_ciphertext.size() - gcm_iv_size - tag_size) = 0;
    // Stop right before the tag
    EVP_DecryptUpdate(
        ctx, outbuf, &outlen,
        (unsigned char *)gcm_ciphertext.data() + gcm_iv_size + tag_size,
        gcm_ciphertext.size() - gcm_iv_size - tag_size);

    /* Set expected tag value. Works in OpenSSL 1.0.1d and later */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, tag_size, (void *)gcm_tag);

    /* Finalise: note get no output for GCM */
    int tmplen;
    int rv = EVP_DecryptFinal_ex(ctx, outbuf + outlen, &tmplen);

    if (rv <= 0) {
        EVP_CIPHER_CTX_free(ctx);
        return "";
    }
    std::string gcm_plaintext = std::string((char *)outbuf, outlen);
    EVP_CIPHER_CTX_free(ctx);
    free(outbuf);
    free(gcm_iv);
    return gcm_plaintext;
}

bool AesGcmEncryptor::AesGcmEncryptFile(
    const std::string &gcm_plaintext_file, const int step_size,
    const std::string &gcm_key_base64, int gcm_key_size,
    const std::string &gcm_ciphertext_file) {
    const int tag_size = 16;
    const int gcm_iv_size = 12;
    if (gcm_plaintext_file.empty() || gcm_ciphertext_file.empty() || gcm_key_base64.empty()) {
        return false;
    }
    srand((int)time(NULL));
    std::string key = CodecUtil::Base64Decode(gcm_key_base64);
    if (key.length() * 8 != gcm_key_size) {
        return false;
    }
    if (256 != gcm_key_size && 128 != gcm_key_size) {
        return false;
    }
    EVP_CIPHER_CTX *ctx;
    unsigned char *outbuf = (unsigned char *)malloc(step_size);
    unsigned char *step_buf = (unsigned char *)malloc(step_size);
    unsigned char *gcm_tag = (unsigned char *)malloc(tag_size);
    unsigned char *gcm_iv = (unsigned char *)malloc(gcm_iv_size);
    for (int i = 0; i < gcm_iv_size; ++i) {
        gcm_iv[i] = rand() % 256;
    }
    std::ifstream in_file_io(gcm_plaintext_file,
                             std::ios::in | std::ios::binary);
    if (!in_file_io.good()) {
        return false;
    }
    int inlen = 0, outlen = 0;
    std::ofstream out_file_io(gcm_ciphertext_file,
                              std::ios::out | std::ios::binary);
    out_file_io.write((char *)gcm_iv, gcm_iv_size);
    out_file_io.write((char *)gcm_tag, tag_size);

    ctx = EVP_CIPHER_CTX_new();
    /* Set cipher type and mode */
    if (gcm_key_size == 256) {
        EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    } else {
        EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    /* Set IV length if default 96 bits is not appropriate */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, gcm_iv_size, NULL);
    int count = 0;
    while (!in_file_io.eof()) {
        in_file_io.read((char *)step_buf, step_size);
        inlen = in_file_io.gcount();
        outlen = 0;
        EVP_EncryptUpdate(ctx, outbuf, &outlen, step_buf, inlen);
        out_file_io.write((char *)outbuf, outlen);
        count++;
        if (count % AdvcSysConfig::GetEncryptSleepCount() == 0) {
            usleep(AdvcSysConfig::GetEncryptSleepTimeInMicrosecond());
        }
    }
    EVP_EncryptFinal_ex(ctx, outbuf, &outlen);
    // /* Get tag */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, tag_size, (void *)gcm_tag);
    // /* Output tag */
    // printf("Tag: outlen:%d\n", *outlen);
    // BIO_dump_fp(stdout, (const char*)(*outbuf), 16);
    out_file_io.seekp(gcm_iv_size, std::ios::beg);
    out_file_io.write((char *)gcm_tag, tag_size);

    EVP_CIPHER_CTX_free(ctx);
    free(outbuf);
    free(step_buf);
    free(gcm_iv);
    free(gcm_tag);
    in_file_io.close();
    out_file_io.close();
    return true;
}

bool AesGcmEncryptor::AesGcmDecryptFile(
    const std::string &gcm_ciphertext_file, const int step_size,
    const std::string &gcm_key_base64, int gcm_key_size,
    const std::string &gcm_plaintext_file) {
    const int tag_size = 16;
    const int gcm_iv_size = 12;
    if (gcm_plaintext_file.empty() || gcm_ciphertext_file.empty() || gcm_key_base64.empty()) {
        return false;
    }
    if (256 != gcm_key_size && 128 != gcm_key_size) {
        return false;
    }
    std::string key = CodecUtil::Base64Decode(gcm_key_base64);
    if (key.length() * 8 != gcm_key_size) {
        return false;
    }

    unsigned char *outbuf = (unsigned char *)malloc(step_size);
    unsigned char *step_buf = (unsigned char *)malloc(step_size);
    unsigned char *gcm_tag = (unsigned char *)malloc(tag_size);
    unsigned char *gcm_iv = (unsigned char *)malloc(gcm_iv_size);

    std::ifstream in_file_io(gcm_ciphertext_file,
                             std::ios::in | std::ios::binary);
    if (!in_file_io.good()) {
        return false;
    }
    in_file_io.read((char *)gcm_iv, gcm_iv_size);
    in_file_io.read((char *)gcm_tag, tag_size);

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    /* Select cipher */
    if (gcm_key_size == 256) {
        EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    } else {
        EVP_DecryptInit_ex(ctx, EVP_aes_128_gcm(), NULL,
                           (unsigned char *)key.data(), gcm_iv);
    }
    EVP_CIPHER_CTX_set_padding(ctx, 0);
    /* Set IV length, omit for 96 bits */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, gcm_iv_size, NULL);

    std::ofstream out_file_io(gcm_plaintext_file,
                              std::ios::out | std::ios::binary);
    int inlen = 0, outlen = 0;
    int count = 0;
    while (!in_file_io.eof()) {
        in_file_io.read((char *)step_buf, step_size);
        inlen = in_file_io.gcount();
        outlen = 0;
        EVP_DecryptUpdate(ctx, outbuf, &outlen, step_buf, inlen);
        out_file_io.write((char *)outbuf, outlen);
        count++;
        if (count % AdvcSysConfig::GetDecryptSleepCount() == 0) {
            usleep(AdvcSysConfig::GetDecryptSleepTimeInMicrosecond());
        }
    }
    /* Set expected tag value. Works in OpenSSL 1.0.1d and later */
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, tag_size, (void *)gcm_tag);

    /* Finalise: note get no output for GCM */
    int tmplen;
    int rv = EVP_DecryptFinal_ex(ctx, outbuf, &tmplen);
    if (rv <= 0) {
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }
    EVP_CIPHER_CTX_free(ctx);
    free(outbuf);
    free(step_buf);
    free(gcm_iv);
    free(gcm_tag);
    return true;
}
}  // namespace advc