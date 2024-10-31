/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2024. All rights reserved.
 * Description: ota_download.cpp is designed for https.
 */

#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <sys/stat.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/pkcs7.h>

#include "ota_download.h"
#include "json/json.h"
#include "rsa_pss.h"
#include "http_client.h"
#include "log_moudle_init.h"
#if defined(BUILD_FOR_ORIN) || defined(BUILD_FOR_X86)
#include "lrunzip.h"
#endif

#if defined(BUILD_FOR_ORIN) || defined(BUILD_FOR_X86)
#define DIGEST_MATH "SHA512"
const EVP_MD* md = EVP_sha512();
#else
#define DIGEST_MATH "SHA256"
const EVP_MD* md = EVP_sha256();
#endif

#define OPENSSL_3_0 0

namespace hozon {
namespace netaos {
namespace https {
using namespace hozon::netaos::crypto;
const std::uint32_t MAX_SIGN_VALUE_FILE_SIZE = 16 * 1024;
const std::uint32_t READ_FILE_BUFF_SIZE = 1024;
std::atomic_bool  OtaDownload::initFlag_ = false;


void OtaDownload::Destroy() {
    HTTPS_INFO << "OtaDownload Destroy";
    if (http_client_) {
        http_client_ = nullptr;
    }
}


void OtaDownload::Init() {
    std::lock_guard<std::mutex> lock(init_mutex_);
    if (initFlag_) {
        return;
    }
    http_client_ = std::make_shared<HttpClient>();
    InitParam();
    if (http_client_) {
          http_client_->Init();
          http_client_->Start();
    } else {
        HTTPS_ERROR << "http_client_ is null";
    }
    initFlag_ = true;
}

OtaDownload::OtaDownload() {
    if (LogModuleInit::getInstance()) {
        LogModuleInit::getInstance()->initLog();
    }
}

int OtaDownload::Download(RequestPtr req_ptr, ResponseHandler handler) {
    HTTPS_INFO << "OtaDownload Download";
    if (http_client_) {
        req_id_ = http_client_->HttpRequest(req_ptr, handler);
        return req_id_;
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return -1;
    }
}

bool OtaDownload::StopDownLoad() {
    HTTPS_INFO << "OtaDownload StopDownLoad";
    if (http_client_) {
        http_client_->Stop();
        return true;
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return false;
    }
}

bool OtaDownload::ReStartDownLoad() {
    HTTPS_INFO << "OtaDownload ReStartDownLoad";
    if (http_client_) {
        http_client_->ReStart();
        return true;
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return false;
    }
}

bool OtaDownload::CancelDownLoad() {
    HTTPS_INFO << "OtaDownload CancelDownLoad";
    if (http_client_ && (req_id_ > 0)) {
        http_client_->CancelRequest(req_id_);
        initFlag_ = false;
        return true;
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return false;
    }
  return false;
}

bool OtaDownload::QueryDownloadInfo(std::vector<Response>& respInfo) {
    HTTPS_INFO << "OtaDownload QueryDownloadInfo";
    if (http_client_) {
        http_client_->Query(respInfo);
        return true;
    } else {
        HTTPS_ERROR << "http_client_ is null";
        return false;
    }
}

// todo: 从savepath 和 url 拼接待解析的文件路径
std::string OtaDownload::GetVerifyFilePath() { return ""; }

std::string OtaDownload::HexToString(const char* bytes, int size) {
  char const hex[16] = {'0', '1', '2', '3', '4', '5', '6', '7',
                        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

  std::string str;
  for (int i = 0; i < size; ++i) {
    const char ch = bytes[i];
    str.append(&hex[(ch & 0xF0) >> 4], 1);
    str.append(&hex[ch & 0xF], 1);
  }
  return str;
}

int OtaDownload::StringToHex(const char* bytes, int size,
                                   std::unique_ptr<char[]>& uptr, int* outlen) {
  int len = 0;
  for (int i = 0; i + 1 < size; i += 2) {
    char chl = bytes[i];
    char chr = bytes[i + 1];
    chl = (chl < 'A') ? chl & 0xF : (chl & 0x7) + 9;
    chr = (chr < 'A') ? chr & 0xF : (chr & 0x7) + 9;
    uptr[len] = (chl << 4) + chr;
    len++;
  }
  uptr[len] = 0;
  *outlen = len;
  return len;
}

bool OtaDownload::Unzip(std::string file_path, std::string unzip_path) {
    #if defined(BUILD_FOR_ORIN) || defined(BUILD_FOR_X86)
        int ret = lrunzip(const_cast<char*>(file_path.c_str()), const_cast<char*>(unzip_path.c_str()));
        if (0 == ret) {
            return true;
        } else {
            HTTPS_ERROR << "lrzip fail, ret: "<< ret;
            return false;
        }
    #else
        std::string zip_file_name = file_path;
        std::string cmdstr = "unzip -o -d " + unzip_path + " " + zip_file_name;
        return (system(cmdstr.c_str()) == 0);
    #endif
}

bool OtaDownload::VerifySignByJson(std::string file_path,
                                         std::string root_path) {
  bool result = true;
  struct stat tmp_stat;
  std::string json_file_name = file_path;
  std::string json_sign_file_name =
      file_path.substr(0, file_path.rfind('/')) + "/file_list-json.sign";
  if ((stat(json_file_name.c_str(), &tmp_stat) != 0) ||
      (stat(json_sign_file_name.c_str(), &tmp_stat) != 0)) {
    HTTPS_ERROR << "json file not exsit, not need to verify.";
    return result;
  }

  if (!VerifySign(json_file_name, json_sign_file_name)) {
    HTTPS_ERROR << "verifySign fail.";
    return false;
  }

  if (!VerifyHashByJson(json_file_name, root_path)) {
    HTTPS_ERROR << "verify hash fail.";
    return false;
  }

  return result;
}

bool OtaDownload::VerifyHashByJson(std::string file_path,
                                         std::string root_path) {
  struct stat tmp_stat;
  if (stat(file_path.c_str(), &tmp_stat) != 0) {
    HTTPS_ERROR << "json file not exsit, not need to verify." << file_path;
    return true;
  }

  std::ifstream ifd;
  ifd.open(file_path.c_str(), std::ios::in);
  Json::CharReaderBuilder reader_builder;
  reader_builder["emitUTF8"] = true;
  Json::Value root;
  std::string err_str;
  bool ret = Json::parseFromStream(reader_builder, ifd, &root, &err_str);
  if (!ret) {
    ifd.close();
    HTTPS_ERROR << "json file parseFromStream fail.";
    return false;
  }
  ifd.close();

  Json::Value file_list = root.get("fileList", "null");
  for (const auto& it: file_list) {
    if (!it.isObject() || !it.getMemberNames().size()) {
      HTTPS_ERROR << "json value is not correct.";
      return false;
    }
    std::string path = it.getMemberNames().at(0);
    std::string hash = it[path].asString();
    if (!VerifyHash(root_path + "/" + path, hash, DIGEST_MATH)) {
      HTTPS_ERROR << "verify file hash fail. file: " << root_path + "/" + path
               << " hash: " << hash << " digest method: " << DIGEST_MATH;
      return false;
    }
  }

  return true;
}

bool OtaDownload::VerifyHash(std::string file_path,
                                   std::string hash_value,
                                   std::string digest_meth) {
#if OPENSSL_3_0 //openssl3.0
  bool result = false;
  unsigned char buffer[READ_FILE_BUFF_SIZE];
  size_t digest_size;
  BIO* reading;
  std::string calc_value_str;

  std::unique_ptr<BIO, void (*)(BIO*)> uptr_input(
      BIO_new_file(file_path.c_str(), "r"), [](BIO* p) { BIO_free(p); });
  if (uptr_input.get() == NULL) {
    HTTPS_ERROR << "MakeLocalCsrSSLApi BIO_new_file err " << file_path.c_str();
    ERR_print_errors_fp(stderr);
    return result;
  }

  std::unique_ptr<OSSL_LIB_CTX, void (*)(OSSL_LIB_CTX*)> uptr_library_context(
      OSSL_LIB_CTX_new(), [](OSSL_LIB_CTX* p) { OSSL_LIB_CTX_free(p); });
  if (uptr_library_context.get() == NULL) {
    HTTPS_ERROR << "OSSL_LIB_CTX_new() returned NULL.";
    ERR_print_errors_fp(stderr);
    return result;
  }

  /*
   * Fetch a message digest by name
   * The algorithm name is case insensitive.
   * See providers(7) for details about algorithm fetching
   */

  std::unique_ptr<EVP_MD, void (*)(EVP_MD*)> uptr_md(
      EVP_MD_fetch(uptr_library_context.get(), digest_meth.c_str(), NULL),
      [](EVP_MD* p) { EVP_MD_free(p); });
  if (uptr_md.get() == NULL) {
    HTTPS_ERROR << "EVP_MD_fetch did not find " << digest_meth;
    ERR_print_errors_fp(stderr);
    return result;
  }
  digest_size = EVP_MD_get_size(uptr_md.get());

  std::unique_ptr<char[], void (*)(char*)> uptr_digest_value(
      reinterpret_cast<char*>(OPENSSL_malloc(digest_size)),
      [](char* p) { OPENSSL_free(p); });
  if (uptr_digest_value.get() == NULL) {
    HTTPS_ERROR << "Can't allocate " << digest_size
             << " bytes for the digest value";
    ERR_print_errors_fp(stderr);
    return result;
  }
  /* Make a bio that uses the digest */

  std::unique_ptr<BIO, void (*)(BIO*)> uptr_bio_digest(
      BIO_new(BIO_f_md()), [](BIO* p) { BIO_free(p); });
  if (uptr_bio_digest.get() == NULL) {
    HTTPS_ERROR << "BIO_new(BIO_f_md()) returned NULL.";
    ERR_print_errors_fp(stderr);
    return result;
  }

  /* set our bio_digest BIO to digest data */
  if (BIO_set_md(uptr_bio_digest.get(), uptr_md.get()) != 1) {
    HTTPS_ERROR << "BIO_set_md failed.";
    ERR_print_errors_fp(stderr);
    return result;
  }
  /*-
   * We will use BIO chaining so that as we read, the digest gets updated
   * See the man page for BIO_push
   */

  reading = BIO_push(uptr_bio_digest.get(), uptr_input.get());

  while (BIO_read(reading, buffer, sizeof(buffer)) > 0) {
  }

  /*-
   * BIO_gets must be used to calculate the final
   * digest value and then copy it to digest_value.
   */

  if (BIO_gets(uptr_bio_digest.get(), uptr_digest_value.get(), digest_size) !=
      static_cast<int>(digest_size)) {
    HTTPS_ERROR << "BIO_gets(bio_digest) failed.";
    ERR_print_errors_fp(stderr);
    return result;
  }

  calc_value_str = HexToString(uptr_digest_value.get(), digest_size);
  HTTPS_DEBUG << "calc hash str: " << calc_value_str;
  HTTPS_DEBUG << "input hash str: " << hash_value;
  if (!hash_value.compare(calc_value_str)) {
    result = true;
  }

  if (!result) ERR_print_errors_fp(stderr);

  return result;
#else
  bool result = false;
  unsigned char buffer[READ_FILE_BUFF_SIZE];
  size_t digest_size;
  BIO* reading;
  std::string calc_value_str;

  std::unique_ptr<BIO, void (*)(BIO*)> uptr_input(
      BIO_new_file(file_path.c_str(), "r"), [](BIO* p) { BIO_free(p); });
  if (uptr_input.get() == NULL) {
      HTTPS_ERROR << "Failed to open file: " << file_path;
      ERR_print_errors_fp(stderr);
      return result;
  }

  /* Fetch a message digest by name */
  const EVP_MD* md = EVP_get_digestbyname(digest_meth.c_str());
  if (md == NULL) {
      HTTPS_ERROR << "Unsupported digest method: " << digest_meth;
      return result;
  }
  digest_size = EVP_MD_size(md);

  std::unique_ptr<unsigned char[], void (*)(unsigned char*)> uptr_digest_value(
      reinterpret_cast<unsigned char*>(OPENSSL_malloc(digest_size)),
      [](unsigned char* p) { OPENSSL_free(p); });
  if (uptr_digest_value.get() == NULL) {
      HTTPS_ERROR << "Can't allocate " << digest_size
                << " bytes for the digest value";
      ERR_print_errors_fp(stderr);
      return result;
  }

  /* Make a bio that uses the digest */
  std::unique_ptr<BIO, void (*)(BIO*)> uptr_bio_digest(
      BIO_new(BIO_f_md()), [](BIO* p) { BIO_free(p); });
  if (uptr_bio_digest.get() == NULL) {
      HTTPS_ERROR << "Failed to create BIO object";
      ERR_print_errors_fp(stderr);
      return result;
  }

  /* Set our bio_digest BIO to digest data */
  if (BIO_set_md(uptr_bio_digest.get(), md) != 1) {
      HTTPS_ERROR << "Failed to set digest method";
      ERR_print_errors_fp(stderr);
      return result;
  }

  /*-
    * We will use BIO chaining so that as we read, the digest gets updated
    * See the man page for BIO_push
    */
  reading = BIO_push(uptr_bio_digest.get(), uptr_input.get());

  while (BIO_read(reading, buffer, sizeof(buffer)) > 0) {
  }

  /*-
    * BIO_gets must be used to calculate the final
    * digest value and then copy it to digest_value.
    */
  if (BIO_gets(uptr_bio_digest.get(), reinterpret_cast<char*>(uptr_digest_value.get()), digest_size) !=
      static_cast<int>(digest_size)) {
      HTTPS_ERROR << "Failed to calculate digest value";
      ERR_print_errors_fp(stderr);
      return result;
  }

  calc_value_str = HexToString(reinterpret_cast<const char*>(uptr_digest_value.get()), digest_size);
  HTTPS_INFO << "Calculated hash value: " << calc_value_str;
  HTTPS_INFO << "Expected hash value: " << hash_value;

  if (calc_value_str == hash_value) {
      result = true;
  }

  if (!result) {
      ERR_print_errors_fp(stderr);
  }

  return result;
#endif
  return false;
}

bool OtaDownload::VerifySign(std::string file_path,
                                   std::string sign_value_file) {
#if OPENSSL_3_0 //openssl3.0
  bool result = false;
  std::ifstream ifs;
  unsigned int hash_len;
  unsigned char buffer[READ_FILE_BUFF_SIZE];

  std::unique_ptr<FILE, void(*)(FILE*)> uptr_p7_file(
    fopen(sign_value_file.c_str(), "rb"), [](FILE* p){ fclose(p); });
  if (!uptr_p7_file) {
    HTTPS_ERROR << "Failed to open sign value file";
    return result;
  }
  std::unique_ptr<PKCS7, void(*)(PKCS7*)> uptr_p7b(
    PEM_read_PKCS7(uptr_p7_file.get(), NULL, NULL, NULL),
    [](PKCS7* p){ PKCS7_free(p); });
  if (!uptr_p7b) {
    HTTPS_ERROR << "Failed to read pkcs7 sign file";
    return result;
  }
  stack_st_PKCS7_SIGNER_INFO* sig_info_stack = PKCS7_get_signer_info(uptr_p7b.get());
  if (!sig_info_stack) {
    HTTPS_ERROR << "Failed to get pkcs7 signer info stack";
    return result;
  }
  PKCS7_SIGNER_INFO* sig_info = sk_PKCS7_SIGNER_INFO_value(sig_info_stack, 0);
  if (!sig_info) {
    HTTPS_ERROR << "Failed to get pkcs7 signer info";
    return result;
  }
  ASN1_OCTET_STRING* enc_digest = sig_info->enc_digest;
  if (!enc_digest) {
    HTTPS_ERROR << "Failed to get enc digest";
    return result;
  }
  HTTPS_DEBUG << "enc_digest size:" << enc_digest->length;
  HTTPS_DEBUG << "enc_digest: "
              << HexToString((const char*)enc_digest->data, enc_digest->length);

  stack_st_X509* cert_stack = PKCS7_get0_signers(uptr_p7b.get(), NULL, 0);
  if (!cert_stack) {
    HTTPS_ERROR << "Failed to get cert stack";
    return result;
  }
  X509* cert = sk_X509_value(cert_stack, 0);
  if (!cert) {
    HTTPS_ERROR << "Failed to get cert X509";
    return result;
  }
  EVP_PKEY* pkey = X509_get_pubkey(cert);
  if (!pkey) {
    HTTPS_ERROR << "Failed to get publickey";
    return result;
  }

  std::unique_ptr<EVP_MD_CTX, void (*)(EVP_MD_CTX*)> uptr_nctx(
      EVP_MD_CTX_new(), [](EVP_MD_CTX* p) { EVP_MD_CTX_free(p); });
  if (uptr_nctx == NULL) {
    HTTPS_ERROR << "Failed to create MD context.";
    return result;
  }
  if (!EVP_DigestInit_ex(uptr_nctx.get(), md, NULL)) {
    HTTPS_ERROR << "Failed to initialize context";
    return result;
  }
  ifs.open(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    HTTPS_ERROR << "Failed to open file " << file_path;
    return result;
  }
  while (!ifs.eof()) {  // 一直读到文件结束
    ifs.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    int readsize = ifs.gcount();
    HTTPS_DEBUG << "readsize:" << readsize;
    if (!EVP_DigestUpdate(uptr_nctx.get(), buffer, readsize)) {
      HTTPS_ERROR << "Failed to hash message into signing context.";
      ifs.close();
      return result;
    }
  }
  ifs.close();
  if (!EVP_DigestFinal_ex(uptr_nctx.get(), buffer, &hash_len)) {
    HTTPS_ERROR << "Failed to hash message.";
    return result;
  }
  std::string hash = HexToString((const char*)buffer, hash_len);
  HTTPS_DEBUG << "hash_len size:" << hash_len;
  HTTPS_DEBUG << "hash size:" << hash.size();
  HTTPS_DEBUG << "file hash: " << hash;

  if (EVP_DigestVerifyInit_ex(uptr_nctx.get(), NULL, DIGEST_MATH, NULL, NULL,
                              pkey, NULL) == 0) {
    HTTPS_ERROR << "Failed to initialize verify context.";
    return result;
  }
  if (EVP_DigestVerifyUpdate(uptr_nctx.get(), hash.c_str(), hash.size()) == 0) {
    HTTPS_ERROR << "Failed to hash message into verify context.";
    return result;
  }
  if (EVP_DigestVerifyFinal(uptr_nctx.get(),
                            (const unsigned char*)enc_digest->data,
                            enc_digest->length) == 0) {
    HTTPS_ERROR << "Failed to verify signature.";
    ERR_print_errors_fp(stderr);
    return result;
  }
  HTTPS_INFO << "Succeed to verify signature.";
  result = true;
  return result;
#else
  bool result = false;
  std::ifstream ifs;
  unsigned int hash_len;
  unsigned char buffer[READ_FILE_BUFF_SIZE];

  std::unique_ptr<FILE, void(*)(FILE*)> uptr_p7_file(
    fopen(sign_value_file.c_str(), "rb"), [](FILE* p){ fclose(p); });
  if (!uptr_p7_file) {
    HTTPS_ERROR << "Failed to open sign value file";
    return result;
  }
  std::unique_ptr<PKCS7, void(*)(PKCS7*)> uptr_p7b(
    PEM_read_PKCS7(uptr_p7_file.get(), NULL, NULL, NULL),
    [](PKCS7* p){ PKCS7_free(p); });
  if (!uptr_p7b) {
    HTTPS_ERROR << "Failed to read pkcs7 sign file";
    return result;
  }
  stack_st_PKCS7_SIGNER_INFO* sig_info_stack = PKCS7_get_signer_info(uptr_p7b.get());
  if (!sig_info_stack) {
    HTTPS_ERROR << "Failed to get pkcs7 signer info stack";
    return result;
  }
  PKCS7_SIGNER_INFO* sig_info = sk_PKCS7_SIGNER_INFO_value(sig_info_stack, 0);
  if (!sig_info) {
    HTTPS_ERROR << "Failed to get pkcs7 signer info";
    return result;
  }
  ASN1_OCTET_STRING* enc_digest = sig_info->enc_digest;
  if (!enc_digest) {
    HTTPS_ERROR << "Failed to get enc digest";
    return result;
  }
  HTTPS_DEBUG << "enc_digest size:" << enc_digest->length;
  HTTPS_DEBUG << "enc_digest: "
              << HexToString((const char*)enc_digest->data, enc_digest->length);

  stack_st_X509* cert_stack = PKCS7_get0_signers(uptr_p7b.get(), NULL, 0);
  if (!cert_stack) {
    HTTPS_ERROR << "Failed to get cert stack";
    return result;
  }
  X509* cert = sk_X509_value(cert_stack, 0);
  if (!cert) {
    HTTPS_ERROR << "Failed to get cert X509";
    return result;
  }
  EVP_PKEY* pkey = X509_get_pubkey(cert);
  if (!pkey) {
    HTTPS_ERROR << "Failed to get publickey";
    return result;
  }

  std::unique_ptr<EVP_MD_CTX, void (*)(EVP_MD_CTX*)> uptr_nctx(
      EVP_MD_CTX_new(), [](EVP_MD_CTX* p) { EVP_MD_CTX_free(p); });
  if (uptr_nctx == NULL) {
    HTTPS_ERROR << "Failed to create MD context.";
    return result;
  }
  if (!EVP_DigestInit(uptr_nctx.get(), md)) {
    HTTPS_ERROR << "Failed to initialize context";
    return result;
  }
  ifs.open(file_path.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    HTTPS_ERROR << "Failed to open file " << file_path;
    return result;
  }
  while (!ifs.eof()) {  // 一直读到文件结束
    ifs.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    int readsize = ifs.gcount();
    HTTPS_DEBUG << "readsize:" << readsize;
    if (!EVP_DigestUpdate(uptr_nctx.get(), buffer, readsize)) {
      HTTPS_ERROR << "Failed to hash message into signing context.";
      ifs.close();
      return result;
    }
  }
  ifs.close();
  if (!EVP_DigestFinal(uptr_nctx.get(), buffer, &hash_len)) {
    HTTPS_ERROR << "Failed to hash message.";
    return result;
  }
  std::string hash = HexToString((const char*)buffer, hash_len);
  HTTPS_DEBUG << "hash_len size:" << hash_len;
  HTTPS_DEBUG << "hash size:" << hash.size();
  HTTPS_DEBUG << "file hash: " << hash;

  if (EVP_DigestVerifyInit(uptr_nctx.get(), NULL, md, NULL, pkey) == 0) {
    HTTPS_ERROR << "Failed to initialize verify context.";
    return result;
  }
  if (EVP_DigestVerifyUpdate(uptr_nctx.get(), hash.c_str(), hash.size()) == 0) {
    HTTPS_ERROR << "Failed to hash message into verify context.";
    return result;
  }
  if (EVP_DigestVerifyFinal(uptr_nctx.get(),
                            (const unsigned char*)enc_digest->data,
                            enc_digest->length) == 0) {
    HTTPS_ERROR << "Failed to verify signature.";
    ERR_print_errors_fp(stderr);
    return result;
  }
  HTTPS_INFO << "Succeed to verify signature.";
  result = true;
  return result;
#endif
  return false;
}

bool OtaDownload::Sign(const std::string file_path,
                             const std::string sign_file_path) {
#if OPENSSL_3_0 //openssl3.0
  bool result = false;
  EVP_PKEY* pkey = NULL;
  EVP_MD_CTX* mctx = NULL;
  const unsigned char* ppriv_key = NULL;
  const char* propq = NULL;

  size_t sig_len = 0;
  unsigned char sig_value[MAX_SIGN_VALUE_FILE_SIZE] = {0};
  std::string sign_str;
  BIO* input = NULL;
  int i = 0;
  unsigned char buffer[512];

  /* Load DER-encoded RSA private key. */
  ppriv_key = rsa_priv_key;
  pkey = d2i_PrivateKey_ex(EVP_PKEY_RSA, NULL, &ppriv_key, sizeof(rsa_priv_key),
                           NULL, propq);
  if (pkey == NULL) {
    HTTPS_ERROR << "Failed to load private key.";
    goto cleanup;
  }

  /* Create MD context used for signing. */
  mctx = EVP_MD_CTX_new();
  if (mctx == NULL) {
    HTTPS_ERROR << "Failed to create MD context.";
    goto cleanup;
  }

  /* Initialize MD context for signing. */
  if (EVP_DigestSignInit_ex(mctx, NULL, "SHA256", NULL, propq, pkey, NULL) ==
      0) {
    HTTPS_ERROR << "Failed to initialize signing context.";
    goto cleanup;
  }

  // read bio from file path
  input = BIO_new_file(file_path.c_str(), "r");
  if (input == NULL) {
    HTTPS_ERROR << "BIO_new_file err " << file_path;
    goto cleanup;
  }

  while ((i = BIO_read(input, buffer, sizeof(buffer))) > 0) {
    /*
     * Feed data to be signed into the algorithm. This may
     * be called multiple times.
     */
    if (EVP_DigestSignUpdate(mctx, buffer, i) == 0) {
      HTTPS_ERROR << "Failed to hash message into signing context.";
      goto cleanup;
    }
  }

  /* Determine signature length. */
  if (EVP_DigestSignFinal(mctx, NULL, &sig_len) == 0) {
    HTTPS_ERROR << "Failed to get signature length.";
    goto cleanup;
  }

  if (sig_len >= MAX_SIGN_VALUE_FILE_SIZE) {
    HTTPS_ERROR << "signature is too large: " << static_cast<int>(sig_len);
    goto cleanup;
  }

  /* Generate signature. */
  if (EVP_DigestSignFinal(mctx, sig_value, &sig_len) == 0) {
    HTTPS_ERROR << "Failed to sign.";
    goto cleanup;
  }

  result = true;
cleanup:
  EVP_MD_CTX_free(mctx);
  EVP_PKEY_free(pkey);

  // save to file
  if (result) {
    std::fstream fs;
    fs.open(sign_file_path, std::ios::out | std::ios::binary);
    sign_str = HexToString((const char*)sig_value, sig_len);
    HTTPS_INFO << "sign value: " << sign_str;
    fs.write((const char*)sig_value, sig_len);
    fs.close();
  }
  return result;
#endif
  return false;
}

OtaDownload::OtaDownload(
    std::map<std::string, std::string> update_param_map) {
  InitParam();
  SetParam(update_param_map);
}

OtaDownload::~OtaDownload() {
    HTTPS_INFO << "~OtaDownload";
    // http_client_ = nullptr;
    if (initFlag_) {
        StopDownLoad();
    }
}

bool OtaDownload::Verify(const std::string file_path) {
  bool result = false;
  std::string zip_file_path = file_path;
  std::string root_path;
  std::vector<std::string> module_list;
  if ("" == file_path) {
    zip_file_path = GetVerifyFilePath();
  }
  struct stat buffer;
  if (stat(zip_file_path.c_str(), &buffer) != 0) {
    HTTPS_ERROR << "zip file not exist: " << zip_file_path;
    return result;
  }
  // 1.校验zip文件MD5完整性
  if (param_map.find("zip_md5") != param_map.end()) {
    std::string md5str = param_map["zip_md5"];
    if (!VerifyHash(zip_file_path, md5str, "MD5")) {
      HTTPS_ERROR << "verify md5 fail: " << md5str;
      return result;
    } else {
      HTTPS_INFO << "verify md5 success: " << md5str;
    }
  }

  // 2.对称解密zip文件
  // todo: 演示没有加密，先不实现

  if (param_map.find("unzip_path") != param_map.end()) {
    root_path = param_map["unzip_path"];
  } else {
    root_path = zip_file_path.substr(0, zip_file_path.rfind('/')) + "/tmp";
  }

  // 判断解压路径是否存在，如果不存在循环创建
  if (!std::filesystem::exists(root_path)) {
    std::filesystem::create_directories(root_path);
  }

  // 3.解压缩zip文件
  if (!Unzip(zip_file_path, root_path)) {
    HTTPS_ERROR << "unzip file fail: " << zip_file_path;
    CleanTmpDir(root_path);
    return result;
  }

  // 4.递归查找json文件
  // for (const auto& file : std::filesystem::directory_iterator(root_path)) {
  //   if (file.is_directory()) {
  //     module_list.push_back(file.path());
  //   }
  // }
  // 4 验证json索引的签名及验证文件完整性
  if (!VerifySignByJson(root_path + "/file_list.json", root_path)) {
    HTTPS_ERROR << "verify root sign fail: " << root_path + "/file_list.json";
    CleanTmpDir(root_path);
    return result;
  }
  std::unique_ptr<DIR, void (*)(DIR*)> uptr_dir(opendir(root_path.c_str()), [](DIR *p){closedir(p);});
  struct dirent* diread;
  if (uptr_dir != nullptr) {
    while ((diread = readdir(uptr_dir.get())) != nullptr) {
      if (strcmp(diread->d_name, ".") != 0 &&
          strcmp(diread->d_name, "..") != 0) {
        if (diread->d_type == 4) {
          module_list.push_back(root_path + "/" + diread->d_name);
        }
      }
    }
  }

  // 4.2 验证每个文件的完整性
  for (auto file : module_list) {
    if (!VerifySignByJson(file + "/file_list.json", root_path)) {
      HTTPS_ERROR << "verify file sign fail: " << file + "/file_list.json";
      CleanTmpDir(root_path);
      return result;
    }
  }
  result = true;
  // 校验成功不删除root_path
  // CleanTmpDir(root_path);
  HTTPS_INFO << "verify all file success.";
  return result;
  return false;
}

bool OtaDownload::SetParam(
    const std::map<std::string, std::string> update_param_map) {
  std::map<std::string, std::string> new_param = update_param_map;
  std::map<std::string, std::string> old_param = param_map;
  new_param.insert(old_param.begin(), old_param.end());
  param_map = new_param;
  HTTPS_INFO << "update param:";
  for (auto item : param_map) {
    HTTPS_INFO << item.first << "=" << item.second;
  }
  return true;
}

bool OtaDownload::CleanTmpDir(std::string root_path) {
  std::string cmdstr = "rm -rf " + root_path;
  if (0 != system(cmdstr.c_str())) {
    HTTPS_WARN << "clean verify tmp dir: " << root_path
            << " fail. please clean it manual.";
    return false;
  }
  HTTPS_INFO << "clean verify tmp dir: " << root_path << " success.";
  return true;
}

void OtaDownload::InitParam() {}

}  // namespace https
}  // namespace netaos
}  // namespace hozon
