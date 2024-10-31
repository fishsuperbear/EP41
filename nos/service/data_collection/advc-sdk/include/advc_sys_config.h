#ifndef ADVC_SYS_CONF_H
#define ADVC_SYS_CONF_H

#include <stdint.h>

#include "advc_defines.h"
#include "util/log_util.h"

namespace advc {

class AdvcSysConfig {
   public:
    /// \brief 设置文件上传连接超时时间,单位:毫秒
    static void SetUploadConnTimeoutInms(uint64_t time);

    /// \brief 设置文件上传接收超时时间,单位:毫秒
    static void SetUploadRecvTimeoutInms(uint64_t time);

    /// \brief 设置文件上传发送超时时间,单位:毫秒
    static void SetUploadSendTimeoutInms(uint64_t time);

    static uint64_t GetUploadConnTimeoutInms();

    static uint64_t GetUploadRecvTimeoutInms();

    static uint64_t GetUploadSendTimeoutInms();

    /// \brief 设置连接超时时间,单位:毫秒
    static void SetConnTimeoutInms(uint64_t time);

    /// \brief 设置接收超时时间,单位:毫秒
    static void SetRecvTimeoutInms(uint64_t time);

    /// \brief 设置发送超时时间,单位:毫秒
    static void SetSendTimeoutInms(uint64_t time);

    static uint64_t GetConnTimeoutInms();

    static uint64_t GetRecvTimeoutInms();

    static uint64_t GetSendTimeoutInms();

    /// \brief 设置分片上传连接超时时间,单位:毫秒
    static void SetUploadPartConnTimeoutInms(uint64_t time);

    /// \brief 设置分片上传接收超时时间,单位:毫秒
    static void SetUploadPartRecvTimeoutInms(uint64_t time);

    /// \brief 设置分片上传发送超时时间,单位:毫秒
    static void SetUploadPartSendTimeoutInms(uint64_t time);

    static uint64_t GetUploadPartConnTimeoutInms();

    static uint64_t GetUploadPartRecvTimeoutInms();

    static uint64_t GetUploadPartSendTimeoutInms();

    /// \brief 获取日志输出类型,默认输出到屏幕
    static int GetLogOutType();

    /// \brief 获取日志输出等级
    static int GetLogLevel();

    /// \brief 打印AdvcSysConfig的配置详情
    static void PrintValue();

    static int GetControlApiRetryTime();

    static uint64_t GetMultiUploadPartSize();

    static uint64_t GetMultiUploadMaxPartNum();

    static int GetMultiUploadRetryTime();

    static void SetEncryptSleepCount(uint64_t count);

    static void SetEncryptSleepTimeInMicrosecond(uint64_t time);

    static uint64_t GetEncryptSleepCount();

    static uint64_t GetEncryptSleepTimeInMicrosecond();

    static void SetDecryptSleepCount(uint64_t count);

    static void SetDecryptSleepTimeInMicrosecond(uint64_t time);

    static uint64_t GetDecryptSleepCount();

    static uint64_t GetDecryptSleepTimeInMicrosecond();

   private:
    // 打印日志:0,不打印,1:打印到屏幕,2:打印到syslog
    static LOG_OUT_TYPE m_log_outtype;
    // 日志级别:1: ERR, 2: WARN, 3:INFO, 4:DBG
    static LOG_LEVEL m_log_level;
    // 文件上传连接超时时间(毫秒)
    static uint64_t m_upload_conn_timeout_in_ms;
    // 文件上传接收超时时间(毫秒)
    static uint64_t m_upload_recv_timeout_in_ms;
    // 文件上传发送超时时间(毫秒)
    static uint64_t m_upload_send_timeout_in_ms;
    // Http连接超时时间(毫秒)
    static uint64_t m_conn_timeout_in_ms;
    // Http接收超时时间(毫秒)
    static uint64_t m_recv_timeout_in_ms;
    // Http 发送超时时间(毫秒)
    static uint64_t m_send_timeout_in_ms;
    // OpenAPI 最大重试次数
    static int m_control_api_retry_time;
    // 分片上传的分片大小
    static uint64_t m_multi_upload_part_size;
    // 分片上传的最大分片数量
    static uint64_t m_multi_upload_max_part_num;
    // 分片上传API的最大重试次数
    static int m_multi_upload_retry_time;
    // 分片上传分片时的连接超时时间(毫秒)
    static uint64_t m_multi_upload_part_conn_timeout_in_ms;
    // 分片上传分片时的接收超时时间(毫秒)
    static uint64_t m_multi_upload_part_recv_timeout_in_ms;
    // 分片上传分片时的发送超时时间(毫秒)
    static uint64_t m_multi_upload_part_send_timeout_in_ms;
    // 加密过程中，进入睡眠前执行循环次数
    static uint64_t encrypt_sleep_count;
    // 加密过程中，睡眠时间
    static uint64_t encrypt_sleep_time_in_microsecond;
    // 解密过程中，进入睡眠前执行循环次数
    static uint64_t decrypt_sleep_count;
    // 解密过程中，睡眠时间
    static uint64_t decrypt_sleep_time_in_microsecond;
};

}  // namespace advc
#endif
