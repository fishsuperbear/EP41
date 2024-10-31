#ifndef HTTP_SENDER_H
#define HTTP_SENDER_H
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <string>

namespace advc {

class HttpSender {
   public:
    static int SendRequest(const std::string &http_method,
                           const std::string &url_str,
                           const std::map<std::string, std::string> &req_params,
                           const std::map<std::string, std::string> &req_headers,
                           const std::string &req_body,
                           uint64_t conn_timeout_in_ms,
                           uint64_t recv_timeout_in_ms,
                           uint64_t send_timeout_in_ms,
                           std::map<std::string, std::string> *resp_headers,
                           std::string *resp_body, std::string *err_msg);

    static int SendRequest(const std::string &http_method,
                           const std::string &url_str,
                           const std::map<std::string, std::string> &req_params,
                           const std::map<std::string, std::string> &req_headers,
                           std::istream &is,
                           uint64_t conn_timeout_in_ms,
                           uint64_t recv_timeout_in_ms,
                           uint64_t send_timeout_in_ms,
                           std::map<std::string, std::string> *resp_headers,
                           std::string *resp_body,
                           std::string *err_msg,
                           std::atomic<bool> &interrupt,
                           int64_t offset = -1,
                           int64_t sendSize = -1);

    static int SendRequest(const std::string &http_method,
                           const std::string &url_str,
                           const std::map<std::string, std::string> &req_params,
                           const std::map<std::string, std::string> &req_headers,
                           std::istream &is,
                           uint64_t conn_timeout_in_ms,
                           uint64_t recv_timeout_in_ms,
                           uint64_t send_timeout_in_ms,
                           std::map<std::string, std::string> *resp_headers,
                           std::ostream &resp_stream,
                           std::string *err_msg,
                           std::atomic<bool> &interrupt,
                           int64_t offset = -1,
                           int64_t sendSize = -1);
};

}  // namespace advc
#endif  // HTTP_SENDER_H
