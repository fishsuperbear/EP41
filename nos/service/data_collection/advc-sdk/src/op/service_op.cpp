#include "op/service_op.h"

#include <Poco/File.h>
#include <Poco/FileStream.h>

#include <fstream>

#include "Poco/JSON/Parser.h"
#include "advc_config.h"
#include "advc_sys_config.h"
#include "util/codec_util.h"
#include "util/http_sender.h"
#include "util/string_util.h"
#include "util/time_util.h"

namespace advc {

void ServiceOp::getAesKey(UploadTokenStruct &uploadToken, GetAesKeyResp *resp) {
    std::string req_body = "";
    std::map<std::string, std::string> req_params;
    req_params.insert(std::pair<std::string, std::string>("Action", "GetDataKey"));
    req_params.insert(std::pair<std::string, std::string>("Version", kServiceGetDataKeyVersion20220720));
    req_params.insert(std::pair<std::string, std::string>("CustomVehicleID", uploadToken.customVehicleId));

    AdvcResult ret = commonRequest(uploadToken, req_params, "", "GET");
    resp->responseMetadata = ret.responseMetadata;
    resp->success = ret.IsSuccess();
    if (!resp->success) {
        return;
    }
    GetAesKeyInfo getAesKeyInfo = GetAesKeyInfo::Decode(ret.result);
    resp->Result = getAesKeyInfo;
    return;
}

void ServiceOp::applyForUpload(UploadTokenStruct &uploadToken, ApplyForUploadReq &req, ApplyForUploadResp *resp) {
    auto marshal = req.Marshal();
    std::stringstream jsonStream;
    Poco::JSON::Stringifier::stringify(marshal, jsonStream);
    std::string req_body = StringUtil::subreplace(jsonStream.str(), "\\/", "/");
    std::map<std::string, std::string> req_params;
    req_params.insert(std::pair<std::string, std::string>("Action", "ApplyForUpload"));
    req_params.insert(std::pair<std::string, std::string>("Version", kServiceVersion20220304));

    AdvcResult ret = commonRequest(uploadToken, req_params, req_body, "POST");
    resp->responseMetadata = ret.responseMetadata;
    resp->success = ret.IsSuccess();
    if (!resp->success) {
        return;
    }
    UploadPointInfo uploadPointInfo = UploadPointInfo::Decode(ret.result);
    resp->Result = uploadPointInfo;
    return;
}

void ServiceOp::commitUpload(UploadTokenStruct &uploadToken, CommitUploadReq &req, CommitUploadResp *resp) {
    auto marshal = req.Marshal();
    std::stringstream jsonStream;
    Poco::JSON::Stringifier::stringify(marshal, jsonStream);
    std::string req_body = StringUtil::subreplace(jsonStream.str(), "\\/", "/");
    std::map<std::string, std::string> req_params;
    req_params.insert(std::pair<std::string, std::string>("Action", "CommitUpload"));
    req_params.insert(std::pair<std::string, std::string>("Version", kServiceVersion20220304));
    AdvcResult ret = commonRequest(uploadToken, req_params, req_body, "POST");
    resp->success = ret.IsSuccess();
    resp->responseMetadata = ret.responseMetadata;
}

void ServiceOp::putObject(UploadPointInfo &uploadPointInfo, std::istream &is, PutObjectResp *resp,
                          std::atomic<bool> &interrupt) {
    std::string host = uploadPointInfo.uploadHosts[0];
    std::string certificate = uploadPointInfo.uploadInfo.certificate;
    std::string storeUri = uploadPointInfo.uploadInfo.storeUri;
    std::map<std::string, std::string> req_headers;
    std::map<std::string, std::string> req_params;
    req_headers["Host"] = host;
    req_headers["X-UPOP-Authorization"] = uploadPointInfo.uploadInfo.certificate;
    std::map<std::string, std::string> resp_headers;

    char creqUrl[4096];
    sprintf(creqUrl, "%s://%s:%d/storage/v1/%s",
            kUploadHostScheme.c_str(),
            uploadPointInfo.uploadHosts[0].c_str(),
            atoi(kUploadHostPort.c_str()),
            uploadPointInfo.uploadInfo.storeUri.c_str());

    std::string resp_body;
    std::string err_msg;
    HttpSender::SendRequest(
        "PUT", creqUrl, req_params, req_headers, is,
        AdvcSysConfig::GetUploadConnTimeoutInms(),
        AdvcSysConfig::GetUploadRecvTimeoutInms(),
        AdvcSysConfig::GetUploadSendTimeoutInms(), &resp_headers,
        &resp_body, &err_msg, interrupt);
    if (resp_body.find("ApiVersion") == -1) {
        resp->isSuccess = false;
        resp->error.message = err_msg;
        return;
    }
    PutObjectResp temp = PutObjectResp::Decode(resp_body);
    resp->isSuccess = temp.isSuccess;
    resp->etag = temp.etag;
    resp->error = temp.error;
}

void ServiceOp::InitMultipartUpload(UploadPointInfo &info, InitMultipartUploadResp *resp) {
    std::string host = info.uploadHosts[0];
    std::string certificate = info.uploadInfo.certificate;
    std::string storeUri = info.uploadInfo.storeUri;
    std::map<std::string, std::string> req_headers;
    std::map<std::string, std::string> req_params;
    req_headers["Host"] = host;
    req_headers["X-UPOP-Authorization"] = info.uploadInfo.certificate;
    req_params["uploads"] = "";
    std::map<std::string, std::string> resp_headers;

    char creqUrl[4096];
    sprintf(creqUrl, "%s://%s:%d/storage/v1/%s",
            kUploadHostScheme.c_str(),
            info.uploadHosts[0].c_str(),
            atoi(kUploadHostPort.c_str()),
            info.uploadInfo.storeUri.c_str());

    std::string resp_body;
    std::string err_msg;
    int http_code = 0;
    for (int i = 0; i <= AdvcSysConfig::GetMultiUploadRetryTime(); i++) {
        http_code = HttpSender::SendRequest("POST", creqUrl, req_params, req_headers, "",
                                            AdvcSysConfig::GetConnTimeoutInms(),
                                            AdvcSysConfig::GetRecvTimeoutInms(),
                                            AdvcSysConfig::GetSendTimeoutInms(),
                                            &resp_headers,
                                            &resp_body, &err_msg);
        if (http_code <= 499) {
            break;
        }
    }

    if (resp_body.find("ApiVersion") == -1) {
        resp->isSuccess = false;
        resp->error.message = err_msg;
        return;
    }
    InitMultipartUploadResp temp = InitMultipartUploadResp::Decode(resp_body);
    resp->isSuccess = temp.isSuccess;
    resp->uploadId = temp.uploadId;
    resp->error = temp.error;
}

void ServiceOp::UploadPart(UploadPointInfo &info, const std::string &uploadId, std::istream &is,
                           int64_t partNum, int64_t streamOffset, int64_t partSize, UploadPartResp *resp,
                           std::atomic<bool> &interrupt) {
    std::string host = info.uploadHosts[0];
    std::string certificate = info.uploadInfo.certificate;
    std::string storeUri = info.uploadInfo.storeUri;
    std::map<std::string, std::string> req_headers;
    std::map<std::string, std::string> req_params;
    req_headers["Host"] = host;
    req_headers["X-UPOP-Authorization"] = info.uploadInfo.certificate;
    req_params["partNumber"] = std::to_string(partNum);
    req_params["uploadId"] = uploadId;
    std::map<std::string, std::string> resp_headers;
    char creqUrl[4096];
    sprintf(creqUrl,
            "%s://%s:%d/storage/v1/%s",
            kUploadHostScheme.c_str(),
            info.uploadHosts[0].c_str(),
            std::atoi(kUploadHostPort.c_str()),
            info.uploadInfo.storeUri.c_str());

    std::string resp_body;
    std::string err_msg;
    int http_code = 0;
    for (int i = 0; i <= AdvcSysConfig::GetControlApiRetryTime(); i++) {
        http_code = HttpSender::SendRequest(
            "PUT", creqUrl, req_params, req_headers, is,
            AdvcSysConfig::GetUploadPartConnTimeoutInms(),
            AdvcSysConfig::GetUploadPartRecvTimeoutInms(),
            AdvcSysConfig::GetUploadPartSendTimeoutInms(), &resp_headers,
            &resp_body, &err_msg, interrupt, streamOffset, partSize);
        if (http_code <= 499) {
            break;
        }
    }

    if (resp_body.find("ApiVersion") == -1) {
        resp->isSuccess = false;
        resp->error.message = err_msg;
        return;
    }
    UploadPartResp temp = UploadPartResp::Decode(resp_body);
    resp->isSuccess = temp.isSuccess;
    resp->etag = temp.etag;
    resp->partNumber = temp.partNumber;
    resp->error = temp.error;
}

void ServiceOp::CompleteMultipartUpload(UploadPointInfo &info, CompleteMultipartUploadReq &req,
                                        CompleteMultipartUploadResp *resp) {
    std::string host = info.uploadHosts[0];
    std::string certificate = info.uploadInfo.certificate;
    std::string storeUri = info.uploadInfo.storeUri;
    std::map<std::string, std::string> req_headers;
    std::map<std::string, std::string> req_params;
    req_headers["Host"] = host;
    req_headers["X-UPOP-Authorization"] = info.uploadInfo.certificate;
    req_params["uploadId"] = req.uploadId;
    std::map<std::string, std::string> resp_headers;

    char creqUrl[4096];
    sprintf(creqUrl, "%s://%s:%d/storage/v1/%s",
            kUploadHostScheme.c_str(),
            info.uploadHosts[0].c_str(),
            atoi(kUploadHostPort.c_str()),
            info.uploadInfo.storeUri.c_str());

    std::string resp_body;
    std::string err_msg;
    std::stringstream jsonStream;
    Poco::JSON::Stringifier::stringify(req.Marshal(), jsonStream);
    std::string req_body = StringUtil::subreplace(jsonStream.str(), "\\/", "/");
    int http_code = 0;
    for (int i = 0; i <= AdvcSysConfig::GetMultiUploadRetryTime(); i++) {
        http_code = HttpSender::SendRequest("POST", creqUrl, req_params, req_headers, req_body,
                                            AdvcSysConfig::GetConnTimeoutInms(),
                                            AdvcSysConfig::GetRecvTimeoutInms(),
                                            AdvcSysConfig::GetSendTimeoutInms(),
                                            &resp_headers,
                                            &resp_body, &err_msg);
        if (http_code <= 499) {
            break;
        }
    }

    if (resp_body.find("ApiVersion") == -1) {
        resp->isSuccess = false;
        resp->error.message = err_msg;
        return;
    }
    CompleteMultipartUploadResp temp = CompleteMultipartUploadResp::Decode(resp_body);
    resp->isSuccess = temp.isSuccess;
    resp->uploadId = temp.uploadId;
    resp->error = temp.error;
}

UploadFileResp ServiceOp::UploadFile(UploadFileReq &req, std::atomic<bool> &interrupt) {
    UploadTokenStruct uploadTokenStruct{};
    DecodeUploadToken(CodecUtil::Base64Decode(req.rawUploadToken), uploadTokenStruct);
    ApplyForUploadReq applyReq{};
    {
        applyReq.vehicleId = uploadTokenStruct.vehicleId;
        applyReq.customVehicleId = uploadTokenStruct.customVehicleId;
        applyReq.fileName = req.fileName;
        applyReq.fileSize = req.fileSize;
        applyReq.expire = req.expire;
        applyReq.cMeta.Channel = req.channel;
        applyReq.cMeta.FileType = req.fileType;
        applyReq.cMeta.Mtime = req.mtime;
        applyReq.cMeta.ProtocolId = req.protocolId;
        applyReq.encryptMeta = req.encryptMeta;
        applyReq.encrypted = req.encrypted;
    }
    ApplyForUploadResp applyForUploadResp;
    // 申请上传
    applyForUpload(uploadTokenStruct, applyReq, &applyForUploadResp);
    if (!applyForUploadResp.success) {
        SDK_LOG_ERR("Advc applyUpload:%s", applyForUploadResp.responseMetadata.error.ToJsonString().c_str());
        return {false, applyForUploadResp.responseMetadata.error};
    }
    PutObjectResp putObjectResp;
    // 上传文件
    putObject(applyForUploadResp.Result, *req.is, &putObjectResp, interrupt);
    if (!putObjectResp.isSuccess) {
        SDK_LOG_ERR("Advc PutObjectResp:%s", putObjectResp.error.ToJsonString().c_str());
        return {false, putObjectResp.error};
    }

    CommitUploadReq commitUploadRequest;
    {
        commitUploadRequest.VehicleId = uploadTokenStruct.vehicleId;
        commitUploadRequest.CustomVehicleId = uploadTokenStruct.customVehicleId;
        commitUploadRequest.Md5 = req.md5;
        commitUploadRequest.Certificate = applyForUploadResp.Result.uploadInfo.certificate;
    }
    CommitUploadResp commitUploadResp;
    // 提交上传
    commitUpload(uploadTokenStruct, commitUploadRequest, &commitUploadResp);
    if (!commitUploadResp.success) {
        SDK_LOG_ERR("Advc commitUpload Exception:%s",
                    commitUploadResp.responseMetadata.error.ToJsonString().c_str());
        return {false, commitUploadResp.responseMetadata.error};
    }
    return {true, applyForUploadResp.responseMetadata.error};
}

std::string GenerateUploadKey(const ApplyForUploadReq &applyReq) {
    return applyReq.customVehicleId + "_" + applyReq.fileName;
}

static ResumableUploadCheckPoint getOrInitCheckPointInfo(const std::string &checkpointPath) {
    std::fstream output_fstream;
    output_fstream.open(checkpointPath, std::ios_base::out | std::ios_base::in);
    ResumableUploadCheckPoint ret;
    if (!output_fstream) {
        std::ofstream fout(checkpointPath);
        ret.phase = "empty";
        ret.lastUpdatedTime = Time::time_util->getLocalTime();
        std::stringstream jsonStream;
        try {
            Poco::JSON::Stringifier::stringify(ret.Marshal(), jsonStream);
        } catch (const std::exception &ex) {
            SDK_LOG_ERR("Init checkpoint file failed: %s", ex.what());
            const int result = remove(checkpointPath.c_str());
            if (result != 0) {
                SDK_LOG_ERR("Delete corrupt checkpoint file failed: %s", checkpointPath.c_str());
            }
        }
        fout << jsonStream.str();
        fout.close();
    } else {
        std::ostringstream tmp;
        tmp << output_fstream.rdbuf();
        try {
            ret = ResumableUploadCheckPoint::Decode(tmp.str());
        } catch (const std::exception &ex) {
            SDK_LOG_ERR("Get checkpoint file failed: %s", ex.what());
            const int result = remove(checkpointPath.c_str());
            if (result != 0) {
                SDK_LOG_ERR("Delete checkpoint file failed: %s", checkpointPath.c_str());
            }
        }
    }
    output_fstream.close();
    return ret;
}

bool removeCheckPointFile(std::string &checkpointFilePath) {
    return remove(checkpointFilePath.c_str()) != 0;
}

bool updateCheckPointInfo(std::string &checkpointFilePath, ResumableUploadCheckPoint &point) {
    auto *checkpointFile = new Poco::File(checkpointFilePath);
    if (!(checkpointFile->exists())) {
        SDK_LOG_ERR("open checkpoint file failed:%s", checkpointFilePath.c_str());
        delete checkpointFile;
        return false;
    }
    delete checkpointFile;

    point.lastUpdatedTime = Time::time_util->getLocalTime();
    auto *fileStream = new Poco::FileStream(checkpointFilePath + "_new",
                                            std::ios_base::out | std::ios_base::trunc);
    std::stringstream jsonStream;
    try {
        Poco::JSON::Stringifier::stringify(point.Marshal(), jsonStream);
    } catch (const std::exception &ex) {
        SDK_LOG_ERR("update checkpoint file failed:%s", ex.what());
        if (removeCheckPointFile(checkpointFilePath)) {
            SDK_LOG_ERR("Delete corrupt checkpoint file failed:%s", checkpointFilePath.c_str());
        }
        delete fileStream;
        return false;
    }
    std::string retStr = jsonStream.str();
    fileStream->write(retStr.c_str(), (std::streamsize)retStr.length());
    fileStream->close();
    delete fileStream;
    auto *newCheckpointFile = new Poco::File(checkpointFilePath + "_new");
    if (removeCheckPointFile(checkpointFilePath)) {
        SDK_LOG_ERR("Delete corrupt checkpoint file failed:%s", checkpointFilePath.c_str());
    }
    newCheckpointFile->moveTo(checkpointFilePath);
    delete newCheckpointFile;
    return true;
}

UploadFileResp
ServiceOp::ResumableUploadFile(UploadFileReq &req, std::string &checkpointDirPath, std::atomic<bool> &interrupt) {
    UploadTokenStruct uploadTokenStruct{};
    DecodeUploadToken(CodecUtil::Base64Decode(req.rawUploadToken), uploadTokenStruct);

    // 校验文件大小
    std::streampos pos = (*req.is).tellg();
    (*req.is).seekg(0, std::ios::end);
    uint64_t streamSize = (*req.is).tellg();
    (*req.is).seekg(pos);
    if (streamSize != req.fileSize) {
        return {false, ErrorObj(-1, "MisMatchFileSize",
                                "specific file size is not match with the stream size")};
    }

    auto partSize = (int64_t)AdvcSysConfig::GetMultiUploadPartSize();
    int totalPartNum = (int)(req.fileSize / (uint64_t)partSize) + (((int)req.fileSize % partSize) != 0);
    if (totalPartNum > (int64_t)AdvcSysConfig::GetMultiUploadMaxPartNum()) {
        return {false, ErrorObj(-1, "ExceedMaxUploadPartNum",
                                "specific file size is too large to upload")};
    }

    ApplyForUploadReq applyReq{};
    {
        applyReq.vehicleId = uploadTokenStruct.vehicleId;
        applyReq.customVehicleId = uploadTokenStruct.customVehicleId;
        applyReq.fileName = req.fileName;
        applyReq.fileSize = (long)req.fileSize;
        applyReq.expire = (int)req.expire;
        applyReq.cMeta.Channel = req.channel;
        applyReq.cMeta.FileType = req.fileType;
        applyReq.cMeta.Mtime = req.mtime;
        applyReq.cMeta.ProtocolId = req.protocolId;
        applyReq.encryptMeta = req.encryptMeta;
        applyReq.encrypted = req.encrypted;
    }
    std::string uploadKey = GenerateUploadKey(applyReq);
    std::string checkpointFilePath = checkpointDirPath + uploadKey + ".checkpoint";
    ResumableUploadCheckPoint checkPoint = getOrInitCheckPointInfo(checkpointFilePath);
    if (checkPoint.phase.empty()) {
        return {false, ErrorObj(-1, "UnKnownError", "unknown error")};
    }

    // 申请上传
    ApplyForUploadResp applyForUploadResp;
    if (checkPoint.phase == "empty") {
        applyForUpload(uploadTokenStruct, applyReq, &applyForUploadResp);
        if (!applyForUploadResp.success) {
            SDK_LOG_ERR("Advc applyUpload:%s", applyForUploadResp.responseMetadata.error.ToJsonString().c_str());
            return {false, applyForUploadResp.responseMetadata.error};
        }
        checkPoint.uploadPointInfo = applyForUploadResp.Result;
        checkPoint.phase = "applied";
        updateCheckPointInfo(checkpointFilePath, checkPoint);
    } else {
        SDK_LOG_INFO("Advc applyUpload skipped uploadKey:%s", uploadKey.c_str());
    }

    if (req.fileSize <= AdvcSysConfig::GetMultiUploadPartSize()) {
        // 只有一个分片时，使用putObject上传
        PutObjectResp putObjectResp;
        // 上传文件
        if (checkPoint.phase == "applied") {
            putObject(applyForUploadResp.Result, *req.is, &putObjectResp, interrupt);
            if (!putObjectResp.isSuccess) {
                SDK_LOG_ERR("Advc PutObjectResp:%s", putObjectResp.error.ToJsonString().c_str());
                return {false, putObjectResp.error};
            }
            checkPoint.phase = "completed";
            updateCheckPointInfo(checkpointFilePath, checkPoint);
        }
    } else {
        // 初始化分片上传
        InitMultipartUploadResp initMultipartUploadResp;
        if (checkPoint.phase == "applied") {
            InitMultipartUpload(checkPoint.uploadPointInfo, &initMultipartUploadResp);
            if (!initMultipartUploadResp.isSuccess) {
                SDK_LOG_ERR("Advc initMultipartUpload:%s", initMultipartUploadResp.error.ToJsonString().c_str());
                return {false, initMultipartUploadResp.error};
            }
            checkPoint.uploadId = initMultipartUploadResp.uploadId;
            checkPoint.phase = "inited";
            updateCheckPointInfo(checkpointFilePath, checkPoint);
        } else {
            SDK_LOG_INFO("Advc initMultipartUpload skipped uploadKey: %s, uploadId: %s", uploadKey.c_str(),
                         checkPoint.uploadId.c_str());
        }

        // 上传分片
        UploadPartResp uploadPartResp;
        if (checkPoint.phase == "inited" || checkPoint.phase == "uploading") {
            int maxUploadedPartNum = checkPoint.maxUploadedPartNum;
            checkPoint.phase = "uploading";
            updateCheckPointInfo(checkpointFilePath, checkPoint);
            for (int partNum = maxUploadedPartNum + 1; partNum <= totalPartNum; ++partNum) {
                int64_t uploadSize =
                    partNum != totalPartNum ? partSize : (int64_t)req.fileSize - (partNum - 1) * partSize;
                UploadPart(checkPoint.uploadPointInfo, checkPoint.uploadId, *req.is, partNum,
                           (partNum - 1) * partSize,
                           uploadSize, &uploadPartResp, interrupt);
                if (!uploadPartResp.isSuccess) {
                    SDK_LOG_ERR("Advc uploadPart:%s", uploadPartResp.error.ToJsonString().c_str());
                    return {false, uploadPartResp.error};
                }
                checkPoint.maxUploadedPartNum++;
                checkPoint.uploadedParts.emplace_back(uploadPartResp.partNumber, uploadPartResp.etag);
                checkPoint.uploadedSize += uploadSize;
                updateCheckPointInfo(checkpointFilePath, checkPoint);
            }
            checkPoint.phase = "uploaded";
            updateCheckPointInfo(checkpointFilePath, checkPoint);
        } else {
            SDK_LOG_INFO("Advc uploadPart skipped uploadKey: %s, uploadId: %s", uploadKey.c_str(),
                         checkPoint.uploadId.c_str());
        }

        // 完成上传
        CompleteMultipartUploadResp completeMultipartUploadresp;
        if (checkPoint.phase == "uploaded") {
            CompleteMultipartUploadReq multipartUploadReq = {checkPoint.uploadedParts, checkPoint.uploadId};
            CompleteMultipartUpload(checkPoint.uploadPointInfo, multipartUploadReq, &completeMultipartUploadresp);
            if (!completeMultipartUploadresp.isSuccess) {
                SDK_LOG_ERR("Advc completeMultipartUpload: %s",
                            completeMultipartUploadresp.error.ToJsonString().c_str());
                return {false, completeMultipartUploadresp.error};
            }
            checkPoint.phase = "completed";
            updateCheckPointInfo(checkpointFilePath, checkPoint);
        } else {
            SDK_LOG_INFO("Advc uploadPart skipped uploadKey: %s, uploadId: %s", uploadKey.c_str(),
                         checkPoint.uploadId.c_str());
        }
    }

    // 提交上传
    if (checkPoint.phase == "completed") {
        CommitUploadReq commitUploadReq = {
            uploadTokenStruct.vehicleId,
            uploadTokenStruct.customVehicleId,
            req.md5,
            checkPoint.uploadPointInfo.uploadInfo.certificate};
        CommitUploadResp commitUploadResp;
        commitUpload(uploadTokenStruct, commitUploadReq, &commitUploadResp);
        if (!commitUploadResp.success) {
            SDK_LOG_ERR("Advc commitUpload Exception:%s",
                        commitUploadResp.responseMetadata.error.ToJsonString().c_str());
            return {false, commitUploadResp.responseMetadata.error};
        }
        if (removeCheckPointFile(checkpointFilePath)) {
            SDK_LOG_ERR("Advc remove checkpoint file failed:%s", checkpointFilePath.c_str());
        }
    } else {
        SDK_LOG_INFO("Advc commitUpload skipped uploadKey:%s", uploadKey.c_str());
    }
    return {true, applyForUploadResp.responseMetadata.error};
}

}  // namespace advc
