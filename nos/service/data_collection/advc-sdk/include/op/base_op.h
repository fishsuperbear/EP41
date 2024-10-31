#pragma once

#include <cinttypes>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include "advc_config.h"
#include "advc_defines.h"
#include "advc_model.h"
#include "op/advc_result.h"

namespace advc {
class BaseOp {
   public:
    BaseOp() {}

    /// \brief BaseOp析构函数
    ~BaseOp() {}

    SignMetadata GetMetaData();

    std::string SigningKeyV4(std::string secretKey, std::string &date, std::string region, std::string service);

    std::string SignatureV4(std::string &signingKey, const std::string &stringToSign);

    std::string BuildAuthHeaderV4(std::string accessKeyId, std::string &signature, SignMetadata &meta);

    std::string SignRequestStr(UploadTokenStruct &uploadToken, std::map<std::string, std::string> &req_params,
                               std::map<std::string, std::string> &req_headers, std::string httpMethod);

    static void DecodeUploadToken(const std::string &in, UploadTokenStruct &uploadTokenStruct);

    AdvcResult
    commonRequest(UploadTokenStruct &uploadToken, std::map<std::string, std::string> _req_params,
                  const std::string &body, const std::string httpMethod);
};

}  // namespace advc
