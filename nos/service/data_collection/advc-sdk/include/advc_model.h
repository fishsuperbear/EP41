//
// Created by bytedance on 5/18/22.
//

#ifndef ADVC_CPP_SDK_ADVC_MODEL_H
#define ADVC_CPP_SDK_ADVC_MODEL_H

#include <Poco/JSON/Parser.h>

#include <utility>
#include <vector>

#include "advc_defines.h"

namespace advc {

struct SignMetadata {
    std::string algorithm;
    std::string credentialScope;
    std::string date;
    std::string region;
    std::string service;
};

struct ErrorObj {
    int codeN;
    std::string code;
    std::string message;

    ErrorObj() : codeN(0), code(""), message("") {}

    ErrorObj(int codeN, std::string code, std::string message) : codeN(codeN), code(code), message(message) {}

    ErrorObj(std::string message) : codeN(0), code(""), message(message) {}

    std::string ToJsonString() {
        Poco::JSON::Object s;
        if (codeN != 0) {
            s.set("CodeN", codeN);
        }
        if (code != "") {
            s.set("Code", code);
        }
        if (message != "") {
            s.set("Message", message);
        }
        std::stringstream jsonStream;
        Poco::JSON::Stringifier::stringify(s, jsonStream);
        return jsonStream.str();
    }
};

struct ResponseMetadata {
    std::string requestId;
    std::string service;
    std::string region;
    std::string action;
    std::string version;
    ErrorObj error;
};

struct UploadTokenStruct {
    std::string accessKeyId;
    std::string secretAccessKey;
    std::string stsToken;
    std::string vehicleId;
    std::string customVehicleId;
    std::string fileName;
    std::string expiredTime;
    std::string currentTime;
    int TokenQuota;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("AccessKeyId", accessKeyId);
        s.set("SecretAccessKey", secretAccessKey);
        s.set("StsToken", stsToken);
        s.set("VehicleId", vehicleId);
        s.set("CustomVehicleId", customVehicleId);
        s.set("FileName", fileName);
        s.set("TokenQuota", TokenQuota);
        s.set("ExpiredTime", expiredTime);
        s.set("CurrentTime", currentTime);
        return s;
    }
};
struct EncryptMeta {
    std::string AlgorithmMode;
    std::string KeyVersionId;
    std::string EncryptType;
    std::vector<std::string> EncryptedTopics;
    unsigned int StepSize;
    unsigned int KeySize;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("AlgorithmMode", AlgorithmMode);
        s.set("KeyVersionId", KeyVersionId);
        s.set("EncryptType", EncryptType);
        s.set("EncryptedTopics", EncryptedTopics);
        s.set("StepSize", StepSize);
        s.set("KeySize", KeySize);
        return s;
    }
};
class UploadFileReq {
   public:
    UploadFileReq(std::string _upload_token,
                  std::istream *is,
                  const uint64_t _fileSize,
                  std::string _fileName,
                  std::string _md5,
                  const uint64_t _mtime,
                  std::string _fileType,
                  std::string _channel,
                  const uint32_t _expire,
                  std::string _protocolId,
                  bool _encrypted,
                  EncryptMeta _encryptMeta) : rawUploadToken(std::move(_upload_token)),
                                              is(is),
                                              fileSize(_fileSize),
                                              fileName(std::move(_fileName)),
                                              md5(std::move(_md5)),
                                              mtime(_mtime),
                                              fileType(std::move(_fileType)),
                                              channel(std::move(_channel)),
                                              expire(_expire),
                                              protocolId(std::move(_protocolId)),
                                              encrypted(_encrypted),
                                              encryptMeta(_encryptMeta) {
    }

    ~UploadFileReq() = default;

    std::istream *is;
    std::string rawUploadToken;
    uint64_t fileSize = 0;
    std::string fileName;
    std::string md5;
    uint64_t mtime = 0;
    std::string fileType;
    std::string channel;
    uint32_t expire = 0;
    std::string protocolId;
    bool encrypted;
    EncryptMeta encryptMeta;
};

class UploadFileResp {
   public:
    UploadFileResp(bool is_success, ErrorObj error) : is_success(is_success), error(std::move(error)) {}

    bool is_success;
    ErrorObj error;

    virtual ~UploadFileResp() = default;
};

struct CustomMeta {
    std::string Channel = "VEHICLE_UPLOAD";
    std::string FileType = "RAW_BAG";
    unsigned int Mtime;
    std::string ProtocolId;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("Channel", Channel);
        s.set("FileType", FileType);
        s.set("Mtime", Mtime);
        s.set("ProtocolId", ProtocolId);
        return s;
    }
};

struct GetAesKeyReq {
    std::string customVehicleId;
    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("CustomVehicleId", customVehicleId);
        return s;
    }
};

struct GetAesKeyInfo {
    std::string plainText;
    std::string versionId;

    static GetAesKeyInfo Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        GetAesKeyInfo getAesKeyInfo;
        getAesKeyInfo.plainText = ds["PlainText"].toString();
        getAesKeyInfo.versionId = ds["VersionID"].toString();
        return getAesKeyInfo;
    }

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("PlainText", plainText);
        s.set("VersionID", versionId);
        return s;
    }
};

struct GetAesKeyResp {
    bool success{};
    ResponseMetadata responseMetadata;
    GetAesKeyInfo Result;
};

struct ApplyForUploadReq {
    std::string vehicleId;
    std::string customVehicleId;
    std::string fileName;
    long int fileSize;
    int expire;
    CustomMeta cMeta;
    bool encrypted;
    EncryptMeta encryptMeta;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("VehicleId", vehicleId);
        s.set("CustomVehicleId", customVehicleId);
        s.set("FileName", fileName);
        s.set("FileSize", fileSize);
        s.set("Expire", expire);
        s.set("Encrypted", encrypted);
        s.set("CustomMeta", cMeta.Marshal());
        s.set("EncryptMeta", encryptMeta.Marshal());
        return s;
    }
};

struct CommitUploadReq {
    std::string VehicleId;
    std::string CustomVehicleId;
    std::string Md5;
    std::string Certificate;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("VehicleId", VehicleId);
        s.set("CustomVehicleId", CustomVehicleId);
        s.set("Md5", Md5);
        s.set("Certificate", Certificate);
        return s;
    }
};

struct CommitUploadResp {
    bool success{};
    ResponseMetadata responseMetadata;
};

struct UploadInfo {
    std::string storeUri;
    std::string certificate;

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("StoreUri", storeUri);
        s.set("Certificate", certificate);
        return s;
    }
};

struct UploadPointInfo {
    std::vector<std::string> uploadHosts;
    UploadInfo uploadInfo;

    static UploadPointInfo Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        UploadPointInfo uploadPointInfo;
        uploadPointInfo.uploadHosts = std::vector<std::string>();
        uploadPointInfo.uploadHosts.push_back(ds["UploadHosts"][0].toString());
        uploadPointInfo.uploadInfo.certificate = ds["UploadInfo"]["Certificate"].toString();
        uploadPointInfo.uploadInfo.storeUri = ds["UploadInfo"]["StoreUri"].toString();
        return uploadPointInfo;
    }

    Poco::JSON::Object Marshal() {
        Poco::JSON::Object s;
        Poco::JSON::Array array;
        if (!uploadHosts.empty()) {
            array.add(uploadHosts[0]);
        }
        s.set("UploadHosts", array);
        s.set("UploadInfo", uploadInfo.Marshal());
        return s;
    }
};

struct ApplyForUploadResp {
    bool success{};
    ResponseMetadata responseMetadata;
    UploadPointInfo Result;
};

struct PutObjectResp {
    bool isSuccess{};
    std::string etag;
    ErrorObj error;

    static PutObjectResp Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        PutObjectResp putObjectResp;
        if (!ds["Error"].isEmpty()) {
            putObjectResp.isSuccess = false;
            if (!ds["Error"]["Code"].isEmpty()) {
                putObjectResp.error.code = ds["Error"]["Code"].toString();
            }
            if (!ds["Error"]["Message"].isEmpty()) {
                putObjectResp.error.message = ds["Error"]["Message"].toString();
            }
        }
        if (!ds["Data"].isEmpty()) {
            putObjectResp.isSuccess = true;
            putObjectResp.etag = ds["Data"]["Etag"].toString();
        }
        return putObjectResp;
    }
};

struct UploadedPart {
    int PartNumber;
    std::string ETag;

    UploadedPart(int partNumber, std::string eTag) : PartNumber(partNumber), ETag(std::move(eTag)) {}

    Poco::JSON::Object Marshal() const {
        Poco::JSON::Object s;
        s.set("PartNumber", PartNumber);
        s.set("ETag", ETag);
        return s;
    }
};

struct ResumableUploadCheckPoint {
    std::string phase;
    UploadPointInfo uploadPointInfo;
    int maxUploadedPartNum = 0;
    int64_t uploadedSize = 0;
    std::string uploadId;
    std::vector<UploadedPart> uploadedParts;
    long lastUpdatedTime{};

    Poco::JSON::Object Marshal() {
        Poco::JSON::Object s;
        Poco::JSON::Object upi;
        Poco::JSON::Array uploadedPartsArray;
        for (const auto &uploadedPart : uploadedParts) {
            uploadedPartsArray.add(uploadedPart.Marshal());
        }
        s.set("Phase", phase);
        s.set("UploadPointInfo", uploadPointInfo.Marshal());
        s.set("MaxUploadedPartNum", maxUploadedPartNum);
        s.set("UploadedSize", uploadedSize);
        s.set("UploadedParts", uploadedPartsArray);
        s.set("LastUpdatedTime", lastUpdatedTime);
        s.set("UploadId", uploadId);
        return s;
    }

    static ResumableUploadCheckPoint Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        ResumableUploadCheckPoint checkPoint;
        checkPoint.phase = ds["Phase"].toString();
        checkPoint.maxUploadedPartNum = std::atoi(ds["MaxUploadedPartNum"].toString().c_str());
        checkPoint.lastUpdatedTime = std::atoi(ds["LastUpdatedTime"].toString().c_str());
        checkPoint.uploadedSize = std::atoi(ds["UploadedSize"].toString().c_str());
        checkPoint.uploadPointInfo.uploadInfo.storeUri = ds["UploadPointInfo"]["UploadInfo"]["StoreUri"].toString();
        checkPoint.uploadPointInfo.uploadInfo.certificate = ds["UploadPointInfo"]["UploadInfo"]["Certificate"].toString();
        checkPoint.uploadId = ds["UploadId"].toString();
        if (ds["UploadPointInfo"]["UploadHosts"].size() > 0) {
            checkPoint.uploadPointInfo.uploadHosts.push_back(ds["UploadPointInfo"]["UploadHosts"][0].toString());
        }
        if (ds["UploadedParts"].size() != 0) {
            for (int i = 0; i < ds["UploadedParts"].size(); i++) {
                Poco::Dynamic::Var addr = ds["UploadedParts"][i];
                checkPoint.uploadedParts.emplace_back(std::atoi(addr["PartNumber"].toString().c_str()),
                                                      addr["ETag"].toString());
            }
        }
        return checkPoint;
    }
};

struct InitMultipartUploadResp {
    bool isSuccess{};
    std::string uploadId;
    ErrorObj error;

    static InitMultipartUploadResp Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        InitMultipartUploadResp initMultipartUploadResp;
        if (!ds["Error"].isEmpty()) {
            initMultipartUploadResp.isSuccess = false;
            if (!ds["Error"]["Code"].isEmpty()) {
                initMultipartUploadResp.error.code = ds["Error"]["Code"].toString();
            }
            if (!ds["Error"]["Message"].isEmpty()) {
                initMultipartUploadResp.error.message = ds["Error"]["Message"].toString();
            }
        }
        if (!ds["Data"].isEmpty()) {
            initMultipartUploadResp.isSuccess = true;
            initMultipartUploadResp.uploadId = ds["Data"]["UploadId"].toString();
        }
        return initMultipartUploadResp;
    }
};

struct UploadPartResp {
    bool isSuccess{};
    int partNumber{};
    std::string etag;
    ErrorObj error;

    static UploadPartResp Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        UploadPartResp uploadPartResp;
        if (!ds["Error"].isEmpty()) {
            uploadPartResp.isSuccess = false;
            if (!ds["Error"]["Code"].isEmpty()) {
                uploadPartResp.error.code = ds["Error"]["Code"].toString();
            }
            if (!ds["Error"]["Message"].isEmpty()) {
                uploadPartResp.error.message = ds["Error"]["Message"].toString();
            }
        }
        if (!ds["Data"].isEmpty()) {
            uploadPartResp.isSuccess = true;
            uploadPartResp.partNumber = std::atoi(ds["Data"]["PartNumber"].toString().c_str());
            uploadPartResp.etag = ds["Data"]["Etag"].toString();
        }
        return uploadPartResp;
    }
};

struct CompleteMultipartUploadResp {
    bool isSuccess{};
    std::string uploadId;
    ErrorObj error;

    static CompleteMultipartUploadResp Decode(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);

        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        std::string val;
        Poco::DynamicStruct ds = *object;
        CompleteMultipartUploadResp completeMultipartUploadResp;
        if (!ds["Error"].isEmpty()) {
            completeMultipartUploadResp.isSuccess = false;
            if (!ds["Error"]["Code"].isEmpty()) {
                completeMultipartUploadResp.error.code = ds["Error"]["Code"].toString();
            }
            if (!ds["Error"]["Message"].isEmpty()) {
                completeMultipartUploadResp.error.message = ds["Error"]["Message"].toString();
            }
        }
        if (!ds["Data"].isEmpty()) {
            completeMultipartUploadResp.isSuccess = true;
            completeMultipartUploadResp.uploadId = ds["Data"]["UploadId"].toString();
        }
        return completeMultipartUploadResp;
    }
};

struct CompleteMultipartUploadReq {
    std::vector<UploadedPart> uploadedParts;
    std::string uploadId;

    Poco::JSON::Object Marshal() {
        Poco::JSON::Object s;
        Poco::JSON::Array array;

        for (const auto &item : uploadedParts) {
            array.add(item.Marshal());
        }
        s.set("Parts", array);
        return s;
    }
};

}  // namespace advc

#endif  // ADVC_CPP_SDK_ADVC_MODEL_H
