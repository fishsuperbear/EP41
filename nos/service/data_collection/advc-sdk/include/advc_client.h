#ifndef ADVC_API_H
#define ADVC_API_H

#include "op/advc_result.h"
#include "op/service_op.h"
#include "util/auth_tool.h"
#include "util/codec_util.h"
#include "util/time_util.h"

namespace advc {

class AdvcClient {
   public:
    /// \brief AdvcClient构造函数
    ///
    /// \param config    advc配置
    explicit AdvcClient();

    explicit AdvcClient(std::shared_ptr<Time> time_ptr);

    ~AdvcClient();

    /// \brief 获取Aes加密密钥
    GetAesKeyResp GetAesKey(std::string &uploadToken);

    /// \brief 上传文件
    UploadFileResp UploadFile(UploadFileReq &req);

    /// \brief 可中断的上传文件
    UploadFileResp UploadFile(UploadFileReq &req, std::atomic<bool> &interrupt);

    /// \brief 支持断点续传且可中断的文件上传
    UploadFileResp ResumableUploadFile(UploadFileReq &req, std::string &checkpointFilePath,
                                       std::atomic<bool> &interrupt);

    /// \brief 支持断点续传的文件上传
    UploadFileResp ResumableUploadFile(UploadFileReq &req, std::string &checkpointDirPath);

   private:
    static int AdvcInit();

    static void AdvcUInit();

   private:
    ServiceOp m_service_op;  // 内部封装object相关的操作
    static bool s_init;
    static bool s_poco_init;
    static int s_advc_obj_num;
};

}  // namespace advc
#endif
