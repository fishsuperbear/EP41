#include "advc_sys_config.h"

#include <stdint.h>

#include <iostream>

#include "advc_defines.h"

namespace advc {

// 单文件上传超时时间(毫秒)
uint64_t AdvcSysConfig::m_upload_conn_timeout_in_ms = 6 * 1000;
uint64_t AdvcSysConfig::m_upload_recv_timeout_in_ms = 6 * 1000;
uint64_t AdvcSysConfig::m_upload_send_timeout_in_ms = 6 * 1000;

// 普通请求HTTP连接/接收时间(毫秒)
uint64_t AdvcSysConfig::m_conn_timeout_in_ms = 5 * 1000;
uint64_t AdvcSysConfig::m_recv_timeout_in_ms = 5 * 1000;
uint64_t AdvcSysConfig::m_send_timeout_in_ms = 5 * 1000;
int AdvcSysConfig::m_control_api_retry_time = 2;

// 日志输出
LOG_OUT_TYPE AdvcSysConfig::m_log_outtype = ADVC_LOG_STDOUT;
LOG_LEVEL AdvcSysConfig::m_log_level = ADVC_LOG_INFO;
// 分片上传
// 分片大小不能小于5MB
uint64_t AdvcSysConfig::m_multi_upload_part_size = 5 * 1024 * 1024;
uint64_t AdvcSysConfig::m_multi_upload_max_part_num = 2000;

// 分片上传，每个分片的连接超时时间(毫秒)
uint64_t AdvcSysConfig::m_multi_upload_part_conn_timeout_in_ms = 5 * 1000;
// 分片上传，每个分片的接收超时时间(毫秒)
uint64_t AdvcSysConfig::m_multi_upload_part_recv_timeout_in_ms = 5 * 1000;
// 分片上传，每个分片的发送超时时间(毫秒)
uint64_t AdvcSysConfig::m_multi_upload_part_send_timeout_in_ms = 5 * 1000;
// 分片上传重试次数
int AdvcSysConfig::m_multi_upload_retry_time = 1;

// 加密过程中，进入睡眠前执行循环次数
uint64_t AdvcSysConfig::encrypt_sleep_count = 100;
// 加密过程中，睡眠时间（微秒）
uint64_t AdvcSysConfig::encrypt_sleep_time_in_microsecond = 200;

// 解密过程中，进入睡眠前执行循环次数
uint64_t AdvcSysConfig::decrypt_sleep_count = 100;
// 解密过程中，睡眠时间（微秒）
uint64_t AdvcSysConfig::decrypt_sleep_time_in_microsecond = 200;

void AdvcSysConfig::PrintValue() {
    std::cout << "log_outtype:" << m_log_outtype << std::endl;
    std::cout << "log_level:" << m_log_level << std::endl;
    std::cout << "multi_upload_part_size:" << m_multi_upload_part_size << std::endl;
    std::cout << "multi_upload_max_part_num:" << m_multi_upload_max_part_num << std::endl;
    std::cout << "multi_upload_retry_time:" << m_multi_upload_retry_time << std::endl;
    std::cout << "conn_timeout_in_ms:" << m_conn_timeout_in_ms << std::endl;
    std::cout << "recv_timeout_in_ms:" << m_recv_timeout_in_ms << std::endl;
    std::cout << "send_timeout_in_ms:" << m_send_timeout_in_ms << std::endl;
    std::cout << "upload_conn_timeout_in_ms:" << m_upload_conn_timeout_in_ms << std::endl;
    std::cout << "upload_recv_timeout_in_ms:" << m_upload_recv_timeout_in_ms << std::endl;
    std::cout << "upload_send_timeout_in_ms:" << m_upload_send_timeout_in_ms << std::endl;
    std::cout << "multi_upload_part_conn_timeout_in_ms:" << m_multi_upload_part_conn_timeout_in_ms << std::endl;
    std::cout << "multi_upload_part_recv_timeout_in_ms:" << m_multi_upload_part_recv_timeout_in_ms << std::endl;
    std::cout << "multi_upload_part_send_timeout_in_ms:" << m_multi_upload_part_send_timeout_in_ms << std::endl;
    std::cout << "encrypt_sleep_count:" << encrypt_sleep_count << std::endl;
    std::cout << "encrypt_sleep_time_in_microsecond:" << encrypt_sleep_time_in_microsecond << std::endl;
    std::cout << "decrypt_sleep_count:" << decrypt_sleep_count << std::endl;
    std::cout << "decrypt_sleep_time_in_microsecond:" << decrypt_sleep_time_in_microsecond << std::endl;
}
void AdvcSysConfig::SetUploadConnTimeoutInms(uint64_t time) {
    m_upload_conn_timeout_in_ms = time;
}

void AdvcSysConfig::SetUploadRecvTimeoutInms(uint64_t time) {
    m_upload_recv_timeout_in_ms = time;
}

void AdvcSysConfig::SetUploadSendTimeoutInms(uint64_t time) {
    m_upload_send_timeout_in_ms = time;
}

void AdvcSysConfig::SetEncryptSleepCount(uint64_t count) {
    encrypt_sleep_count = count;
}

void AdvcSysConfig::SetEncryptSleepTimeInMicrosecond(uint64_t time) {
    encrypt_sleep_time_in_microsecond = time;
}

void AdvcSysConfig::SetDecryptSleepCount(uint64_t count) {
    decrypt_sleep_count = count;
}

void AdvcSysConfig::SetDecryptSleepTimeInMicrosecond(uint64_t time) {
    decrypt_sleep_time_in_microsecond = time;
}

uint64_t AdvcSysConfig::GetUploadConnTimeoutInms() { return m_upload_conn_timeout_in_ms; }

uint64_t AdvcSysConfig::GetUploadRecvTimeoutInms() { return m_upload_recv_timeout_in_ms; }

uint64_t AdvcSysConfig::GetUploadSendTimeoutInms() { return m_upload_send_timeout_in_ms; }

void AdvcSysConfig::SetConnTimeoutInms(uint64_t time) {
    m_conn_timeout_in_ms = time;
}

void AdvcSysConfig::SetRecvTimeoutInms(uint64_t time) {
    m_recv_timeout_in_ms = time;
}

void AdvcSysConfig::SetSendTimeoutInms(uint64_t time) {
    m_send_timeout_in_ms = time;
}
uint64_t AdvcSysConfig::GetConnTimeoutInms() { return m_conn_timeout_in_ms; }

uint64_t AdvcSysConfig::GetRecvTimeoutInms() { return m_recv_timeout_in_ms; }

uint64_t AdvcSysConfig::GetSendTimeoutInms() { return m_send_timeout_in_ms; }

void AdvcSysConfig::SetUploadPartConnTimeoutInms(uint64_t time) {
    m_multi_upload_part_conn_timeout_in_ms = time;
}

void AdvcSysConfig::SetUploadPartRecvTimeoutInms(uint64_t time) {
    m_multi_upload_part_recv_timeout_in_ms = time;
}

void AdvcSysConfig::SetUploadPartSendTimeoutInms(uint64_t time) {
    m_multi_upload_part_send_timeout_in_ms = time;
}

uint64_t AdvcSysConfig::GetUploadPartConnTimeoutInms() { return m_multi_upload_part_conn_timeout_in_ms; }

uint64_t AdvcSysConfig::GetUploadPartRecvTimeoutInms() { return m_multi_upload_part_recv_timeout_in_ms; }

uint64_t AdvcSysConfig::GetUploadPartSendTimeoutInms() { return m_multi_upload_part_send_timeout_in_ms; }

int AdvcSysConfig::GetLogOutType() { return (int)m_log_outtype; }

int AdvcSysConfig::GetLogLevel() { return (int)m_log_level; }

int AdvcSysConfig::GetControlApiRetryTime() { return m_control_api_retry_time; }

uint64_t AdvcSysConfig::GetMultiUploadPartSize() {
    return m_multi_upload_part_size;
}

uint64_t AdvcSysConfig::GetMultiUploadMaxPartNum() {
    return m_multi_upload_max_part_num;
}
int AdvcSysConfig::GetMultiUploadRetryTime() {
    return m_multi_upload_retry_time;
}

uint64_t AdvcSysConfig::GetEncryptSleepCount() {
    return encrypt_sleep_count;
}

uint64_t AdvcSysConfig::GetEncryptSleepTimeInMicrosecond() {
    return encrypt_sleep_time_in_microsecond;
};

uint64_t AdvcSysConfig::GetDecryptSleepCount() {
    return decrypt_sleep_count;
}

uint64_t AdvcSysConfig::GetDecryptSleepTimeInMicrosecond() {
    return decrypt_sleep_time_in_microsecond;
};

}  // namespace advc
