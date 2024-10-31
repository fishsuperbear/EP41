#include "util/http_sender.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>

#include "Poco/Buffer.h"
#include "Poco/Net/Context.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/Net/HTTPSClientSession.h"
#include "Poco/Net/NetException.h"
#include "Poco/StreamCopier.h"
#include "Poco/URI.h"
#include "advc_defines.h"
#include "advc_sys_config.h"
#include "util/codec_util.h"
#include "util/string_util.h"

namespace advc {
int HttpSender::SendRequest(
    const std::string &http_method, const std::string &url_str,
    const std::map<std::string, std::string> &req_params,
    const std::map<std::string, std::string> &req_headers,
    const std::string &req_body, uint64_t conn_timeout_in_ms,
    uint64_t recv_timeout_in_ms, uint64_t send_timeout_in_ms,
    std::map<std::string, std::string> *resp_headers, std::string *resp_body,
    std::string *err_msg) {
    std::istringstream is(req_body);
    std::ostringstream oss;
    std::atomic<bool> interrupt{};
    int ret = SendRequest(http_method, url_str, req_params, req_headers, is,
                          conn_timeout_in_ms, recv_timeout_in_ms, send_timeout_in_ms, resp_headers,
                          oss, err_msg, interrupt);
    *resp_body = oss.str();
    return ret;
}

int HttpSender::SendRequest(
    const std::string &http_method, const std::string &url_str,
    const std::map<std::string, std::string> &req_params,
    const std::map<std::string, std::string> &req_headers, std::istream &is,
    uint64_t conn_timeout_in_ms, uint64_t recv_timeout_in_ms, uint64_t send_timeout_in_ms,
    std::map<std::string, std::string> *resp_headers, std::string *resp_body,
    std::string *err_msg, std::atomic<bool> &interrupt, int64_t offset, int64_t sendSize) {
    std::ostringstream oss;
    int ret = SendRequest(http_method, url_str, req_params, req_headers, is,
                          conn_timeout_in_ms, recv_timeout_in_ms, send_timeout_in_ms, resp_headers,
                          oss, err_msg, interrupt, offset, sendSize);
    *resp_body = oss.str();
    return ret;
}

int HttpSender::SendRequest(
    const std::string &http_method, const std::string &url_str,
    const std::map<std::string, std::string> &req_params,
    const std::map<std::string, std::string> &req_headers, std::istream &is,
    uint64_t conn_timeout_in_ms, uint64_t recv_timeout_in_ms, uint64_t send_timeout_in_ms,
    std::map<std::string, std::string> *resp_headers, std::ostream &resp_stream,
    std::string *err_msg, std::atomic<bool> &interrupt, int64_t offset, int64_t sendSize) {
    Poco::Net::HTTPResponse res;
    try {
        SDK_LOG_INFO("send request to [%s]", url_str.c_str());
        Poco::URI url(url_str);
        std::unique_ptr<Poco::Net::HTTPClientSession> session;

        if (StringUtil::StringStartsWithIgnoreCase(url_str, "https")) {
            Poco::Net::Context::Ptr context =
                new Poco::Net::Context(Poco::Net::Context::CLIENT_USE, "", "", "",
                                       Poco::Net::Context::VERIFY_RELAXED, 9, true,
                                       "ALL:!ADH:!LOW:!EXP:!MD5:@STRENGTH");
            session.reset(new Poco::Net::HTTPSClientSession(url.getHost(),
                                                            url.getPort(), context));
        } else {
            session.reset(
                new Poco::Net::HTTPClientSession(url.getHost(), url.getPort()));
        }

        session->setTimeout(Poco::Timespan(0, conn_timeout_in_ms * 1000), Poco::Timespan(0, send_timeout_in_ms * 1000), Poco::Timespan(0, recv_timeout_in_ms * 1000));
        // 1. 拼接path_query字符串
        std::string path = url.getPath();
        if (path.empty()) {
            path += "/";
        }

        std::string query_str;
        for (const auto &req_param : req_params) {
            std::string part;
            if (req_param.second.empty()) {
                part = CodecUtil::UrlEncode(req_param.first) + "&";
            } else {
                part = CodecUtil::UrlEncode(req_param.first) + "=" +
                       CodecUtil::UrlEncode(req_param.second) + "&";
            }
            query_str += part;
        }

        if (!query_str.empty()) {
            query_str = "?" + query_str.substr(0, query_str.size() - 1);
        }
        std::string path_and_query_str = CodecUtil::EncodeKey(path) + query_str;

        // 2. 创建http request, 并填充头部
        Poco::Net::HTTPRequest req(http_method, path_and_query_str,
                                   Poco::Net::HTTPMessage::HTTP_1_1);
        for (std::map<std::string, std::string>::const_iterator c_itr =
                 req_headers.begin();
             c_itr != req_headers.end(); ++c_itr) {
            req.add(c_itr->first, c_itr->second);
        }

        std::streamsize restByteNum;
        // 3. 计算长度
        if (sendSize < 0) {
            std::streampos pos = is.tellg();
            is.seekg(0, std::ios::end);
            restByteNum = is.tellg();
            req.setContentLength(restByteNum);
            is.seekg(pos);
        } else {
            req.setContentLength(sendSize);
            restByteNum = sendSize;
            is.seekg(offset);
        }
        std::cout << req.getContentLength();

        std::ostringstream debug_os;
        req.write(debug_os);
        SDK_LOG_DBG("request=[%s]", debug_os.str().c_str());

        // 4. 发送请求
        // 统计上传速率
        std::chrono::time_point<std::chrono::steady_clock> start_ts, end_ts;
        start_ts = std::chrono::steady_clock::now();
        std::ostream &os = session->sendRequest(req);
        Poco::Buffer<char> buffer(4096);
        std::streamsize copy_size = 0;
        std::streamsize readByteNum = restByteNum >= 4096 ? 4096 : restByteNum;
        is.read(buffer.begin(), readByteNum);
        std::streamsize n = is.gcount();

        Poco::Timespan remaining(0, send_timeout_in_ms * 1000);
        Poco::Timestamp start;
        while (n > 0 && restByteNum > 0) {
            if (interrupt) {
                session->reset();
                SDK_LOG_WARN("Outside interrupt request body copy. send_size: %" PRIu64, copy_size);
                *err_msg = "Outside interrupt request body copy. send_size:" + std::to_string(copy_size);
                return -1;
            }
            os.write(buffer.begin(), readByteNum);
            copy_size += readByteNum;
            restByteNum -= readByteNum;
            readByteNum = restByteNum >= 4096 ? 4096 : restByteNum;
            if (is && os) {
                is.read(buffer.begin(), readByteNum);
                n = is.gcount();
            } else {
                n = 0;
            }
            Poco::Timestamp end;
            Poco::Timespan waited(end - start);
            if (waited >= remaining) {
                SDK_LOG_ERR("timeout,cost: %ld ms \n", (end - start) / 1000);
                throw Poco::TimeoutException();
            }
        }
        end_ts = std::chrono::steady_clock::now();
        auto time_consumed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_ts - start_ts)
                .count();
        // 大于100KB才计算速率
        if (time_consumed_ms > 1 && copy_size > 100 * 1024) {
            float rate =
                ((float)copy_size / 1024 / 1024) / ((float)time_consumed_ms / 1000);
            SDK_LOG_DBG("send_size: %" PRIu64 ", time_consumed: %" PRIu64
                        " ms, rate: %.2f MB/s",
                        copy_size, time_consumed_ms, rate);
        }

        // 5. 接收返回
        Poco::Net::StreamSocket &ss = session->socket();
        ss.setReceiveTimeout(Poco::Timespan(0, recv_timeout_in_ms * 1000));
        std::istream &recv_stream = session->receiveResponse(res);

        // 6. 处理返回
        int ret = res.getStatus();
        resp_headers->insert(res.begin(), res.end());
        start_ts = std::chrono::steady_clock::now();
        copy_size = Poco::StreamCopier::copyStream(recv_stream, resp_stream);
        end_ts = std::chrono::steady_clock::now();
        time_consumed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_ts - start_ts)
                .count();
        // 大于100KB计算速率
        if (time_consumed_ms > 1 && copy_size > 100 * 1024) {
            float rate =
                ((float)copy_size / 1024 / 1024) / ((float)time_consumed_ms / 1000);
            SDK_LOG_DBG("send_size: %" PRIu64 ", time_consumed: %" PRIu64
                        " ms, rate: %.2f MB/s",
                        copy_size, time_consumed_ms, rate);
        }

        {
            std::string resp_header_log;
            resp_header_log.append("response header :\n");
            for (std::map<std::string, std::string>::const_iterator itr =
                     resp_headers->begin();
                 itr != resp_headers->end(); ++itr) {
                resp_header_log.append(itr->first + ": " + itr->second + "\n");
            }
            SDK_LOG_DBG("%s", resp_header_log.c_str());
        }

        SDK_LOG_INFO("Send request over, status=%d, reason=%s", res.getStatus(),
                     res.getReason().c_str());
        return ret;
    } catch (Poco::Net::NetException &ex) {
        SDK_LOG_ERR("Net Exception:%s", ex.displayText().c_str());
        *err_msg = "Net Exception:" + ex.displayText();
        return -1;
    } catch (Poco::TimeoutException &ex) {
        SDK_LOG_ERR("TimeoutException:%s", ex.displayText().c_str());
        *err_msg = "TimeoutException:" + ex.displayText();
        return -1;
    } catch (const std::exception &ex) {
        SDK_LOG_ERR("Exception:%s, errno=%d", std::string(ex.what()).c_str(),
                    errno);
        *err_msg = "Exception:" + std::string(ex.what());
        return -1;
    }
}

}  // namespace advc
