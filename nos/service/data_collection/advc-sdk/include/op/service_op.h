#ifndef SERVICE_OP_H
#define SERVICE_OP_H

#include <atomic>

#include "op/advc_result.h"
#include "op/base_op.h"

namespace advc {

/// \brief 封装了Service相关的操作
class ServiceOp : public BaseOp {
   public:
    /// \brief ServiceOp构造函数
    ///
    /// \param advc_conf Advc配置

    ServiceOp() = default;

    /// \brief ServiceOp析构函数
    virtual ~ServiceOp() = default;

    /// \brief 申请Aes加密密钥
    void getAesKey(UploadTokenStruct &uploadToken, GetAesKeyResp *resp);

    /// \brief 申请文件上传
    void applyForUpload(UploadTokenStruct &uploadToken, ApplyForUploadReq &req, ApplyForUploadResp *resp);

    /// \brief 确认上传
    void commitUpload(UploadTokenStruct &uploadToken, CommitUploadReq &req, CommitUploadResp *resp);

    /// \brief 上传文件
    static void putObject(UploadPointInfo &info, std::istream &is, PutObjectResp *resp, std::atomic<bool> &interrupt);

    /// \brief 将的文件上传
    UploadFileResp UploadFile(UploadFileReq &req, std::atomic<bool> &interrupt);

    /// \brief 支持断点续传的文件上传
    UploadFileResp
    ResumableUploadFile(UploadFileReq &req, std::string &checkpointDirPath, std::atomic<bool> &interrupt);

    /// \brief 初始化分片上传
    static void InitMultipartUpload(UploadPointInfo &info, InitMultipartUploadResp *resp);

    /// \brief 分片上传
    static void UploadPart(UploadPointInfo &info, const std::string &uploadId, std::istream &is,
                           int64_t partNum, int64_t streamOffset, int64_t partSize, UploadPartResp *resp, std::atomic<bool> &interrupt);

    /// \brief 完成分片上传
    static void CompleteMultipartUpload(UploadPointInfo &info, CompleteMultipartUploadReq &req,
                                        CompleteMultipartUploadResp *resp);
};

}  // namespace advc
#endif
