/*
 * Copyright (c) hozonauto. 2021-2021. All rights reserved.
 * Description: Http common definition
 */

#ifndef V2C_HTTPLIB_HTTP_TYPES_H
#define V2C_HTTPLIB_HTTP_TYPES_H

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "curl/curl.h"

namespace hozon {
namespace netaos {
namespace https {

// Http method definition.
enum HttpMethod {
    HTTP_GET = 0,
    HTTP_POST,
    HTTP_PUT,
    HTTP_DELETE,
};

enum HttpsResultCode:int32_t{
    HttpsResult_Success = 0,
    HttpsResult_InitError = 1,  // mdc内部错误：cm初始化未成功
    HttpsResult_ComError = 2,   // mdc内部错误：网络通信失败
    HttpsResult_HttpErrorStart = 100,       // http error start.
    HttpsResult_ClientCertInvalid = 0xff01,
    HttpsResult_SystemParamInvalid,
    HttpsResult_Timeout,
    HttpsResult_HttpsComError,
    HttpsResult_OtherError,
    HttpsResult_Cancelled
};

enum EncodeFormat { EncodeFormat_Der = 0, EncodeFormat_Pem };

// Http response code definition.
#define HTTP_CODE_OK 200

// Http header definition.
enum ResponseHeaderType {
    Resp_Accept_Ranges,       //    表明服务器是否支持指定范围请求及哪种类型的分段请求    Accept-Ranges: bytes
    Resp_Age,                 //    从原始服务器到代理缓存形成的估算时间（以秒计，非负）    Age: 12
    Resp_Allow,               //    对某网络资源的有效的请求行为，不允许则返回405    Allow: GET, HEAD
    Resp_Cache_Control,       //    告诉所有的缓存机制是否可以缓存及哪种类型    Cache-Control: no-cache
    Resp_Content_Encoding,    //    web服务器支持的返回内容压缩编码类型。    Content-Encoding: gzip
    Resp_Content_Language,    //    响应体的语言    Content-Language: en,zh
    Resp_Content_Length,      //    响应体的长度    Content-Length: 348
    Resp_Content_Location,    //    请求资源可替代的备用的另一地址    Content-Location: /index.htm
    Resp_Content_MD5,         //    返回资源的MD5校验值    Content-MD5: Q2hlY2sgSW50ZWdyaXR5IQ==
    Resp_Content_Range,       //    在整个返回体中本部分的字节位置    Content-Range: bytes 21010-47021/47022
    Resp_Content_Type,        //    返回内容的MIME类型    Content-Type: text/html; charset=utf-8
    Resp_Date,                //    原始服务器消息发出的时间    Date: Tue, 15 Nov 2010 08:12:31 GMT
    Resp_ETag,                //    请求变量的实体标签的当前值    ETag: “737060cd8c284d8af7ad3082f209582d”
    Resp_Expires,             //    响应过期的日期和时间    Expires: Thu, 01 Dec 2010 16:00:00 GMT
    Resp_Last_Modified,       //    请求资源的最后修改时间    Last-Modified: Tue, 15 Nov 2010 12:45:26 GMT
    Resp_Location,            //    用来重定向接收方到非请求URL的位置来完成请求或标识新的资源    Location: http://www.zcmhi.com/archives/94.html
    Resp_Pragma,              //    包括实现特定的指令，它可应用到响应链上的任何接收方    Pragma: no-cache
    Resp_Proxy_Authenticate,  //    它指出认证方案和可应用到代理的该URL上的参数    Proxy-Authenticate: Basic
    Resp_refresh,  //    应用于重定向或一个新的资源被创造，在5秒之后重定向（由网景提出，被大部分浏览器支持）    Refresh: 5; url=http://www.zcmhi.com/archives/94.html
    Resp_Retry_After,        //    如果实体暂时不可取，通知客户端在指定时间之后再次尝试    Retry-After: 120
    Resp_Server,             //    web服务器软件名称    Server: Apache/1.3.27 (Unix) (Red-Hat/Linux)
    Resp_Set_Cookie,         //    设置Http Cookie    Set-Cookie: UserID=JohnDoe; Max-Age=3600; Version=1
    Resp_Trailer,            //    指出头域在分块传输编码的尾部存在    Trailer: Max-Forwards
    Resp_Transfer_Encoding,  //    文件传输编码    Transfer-Encoding:chunked
    Resp_Vary,               //    告诉下游代理是使用缓存响应还是从原始服务器请求    Vary: *
    Resp_Via,                //    告知代理客户端响应是通过哪里发送的    Via: 1.0 fred, 1.1 nowhere.com (Apache/1.1)
    Resp_Warning,            //    警告实体可能存在的问题    Warning: 199 Miscellaneous warning
    Resp_WWW_Authenticate,   //    表明客户端请求实体应该使用的授权方案    WWW-Authenticate: Basic
};

enum RequestHeaderType {
    Req_Accept,             //    指定客户端能够接收的内容类型    Accept: text/plain, text/html
    Req_Accept_Charset,     //    浏览器可以接受的字符编码集。    Accept-Charset: iso-8859-5
    Req_Accept_Encoding,    //    指定浏览器可以支持的web服务器返回内容压缩编码类型。    Accept-Encoding: compress, gzip
    Req_Accept_Language,    //    浏览器可接受的语言    Accept-Language: en,zh
    Req_Accept_Ranges,      //    可以请求网页实体的一个或者多个子范围字段    Accept-Ranges: bytes
    Req_Authorization,      //    HTTP授权的授权证书    Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
    Req_Cache_Control,      //    指定请求和响应遵循的缓存机制    Cache-Control: no-cache
    Req_Connection,         //    表示是否需要持久连接。（HTTP 1.1默认进行持久连接）    Connection: close
    Req_Cookie,             //    HTTP请求发送时，会把保存在该请求域名下的所有cookie值一起发送给web服务器。    Cookie: $Version=1; Skin=new;
    Req_Content_Length,     //    请求的内容长度    Content-Length: 348
    Req_Content_Type,       //    请求的与实体对应的MIME信息    Content-Type: application/x-www-form-urlencoded
    Req_Date,               //    请求发送的日期和时间    Date: Tue, 15 Nov 2010 08:12:31 GMT
    Req_Expect,             //    请求的特定的服务器行为    Expect: 100-continue
    Req_From,               //    发出请求的用户的Email    From: user@email.com
    Req_Host,               //    指定请求的服务器的域名和端口号    Host: www.zcmhi.com
    Req_If_Match,           //    只有请求内容与实体相匹配才有效    If-Match: “737060cd8c284d8af7ad3082f209582d”
    Req_If_Modified_Since,  //    如果请求的部分在指定时间之后被修改则请求成功，未被修改则返回304代码    If-Modified-Since: Sat, 29 Oct 2010 19:43:31 GMT
    Req_If_None_Match,  //    如果内容未改变返回304代码，参数为服务器先前发送的Etag，与服务器回应的Etag比较判断是否改变    If-None-Match: “737060cd8c284d8af7ad3082f209582d”
    Req_If_Range,             //    如果实体未改变，服务器发送客户端丢失的部分，否则发送整个实体。参数也为Etag    If-Range: “737060cd8c284d8af7ad3082f209582d”
    Req_If_Unmodified_Since,  //    只在实体在指定时间之后未被修改才请求成功    If-Unmodified-Since: Sat, 29 Oct 2010 19:43:31 GMT
    Req_Max_Forwards,         //    限制信息通过代理和网关传送的时间    Max-Forwards: 10
    Req_Pragma,               //    用来包含实现特定的指令    Pragma: no-cache
    Req_Proxy_Authorization,  //    连接到代理的授权证书    Proxy-Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
    Req_Range,                //    只请求实体的一部分，指定范围    Range: bytes=500-999
    Req_Referer,              //    先前网页的地址，当前请求网页紧随其后,即来路    Referer: http://www.zcmhi.com/archives/71.html
    Req_TE,                   //    客户端愿意接受的传输编码，并通知服务器接受接受尾加头信息    TE: trailers,deflate;q=0.5
    Req_Upgrade,              //    向服务器指定某种传输协议以便服务器进行转换（如果支持）    Upgrade: HTTP/2.0, SHTTP/1.3, IRC/6.9, RTA/x11
    Req_User_Agent,           //    User-Agent的内容包含发出请求的用户信息    User-Agent: Mozilla/5.0 (Linux; X11)
    Req_Via,                  //    通知中间网关或代理服务器地址，通信协议    Via: 1.0 fred, 1.1 nowhere.com (Apache/1.1)
    Req_Warning,              //    关于消息实体的警告信息    Warn: 199 Miscellaneous warning
};

#define RESP_HEADER_ACCEPT_RANGES "Accept-Ranges"        // Resp_Accept_Ranges 表明服务器是否支持指定范围请求及哪种类型的分段请求    Accept-Ranges: bytes
#define RESP_HEADER_AGE "Age"                            // Resp_Age 从原始服务器到代理缓存形成的估算时间（以秒计，非负）    Age: 12
#define RESP_HEADER_ALLOW "Allow"                        // Resp_Allow 对某网络资源的有效的请求行为，不允许则返回405    Allow: GET, HEAD
#define RESP_HEADER_CACHE_CONTROL "Cache-Control"        // Resp_Cache_Control 告诉所有的缓存机制是否可以缓存及哪种类型    Cache-Control: no-cache
#define RESP_HEADER_CONTENT_ENCODING "Content-Encoding"  // Resp_Content_Encoding web服务器支持的返回内容压缩编码类型。    Content-Encoding: gzip
#define RESP_HEADER_CONTENT_LANGUAGE "Content-Language"  // Resp_Content_Language 响应体的语言    Content-Language: en,zh
#define RESP_HEADER_CONTENT_LENGTH "Content-Length"      // Resp_Content_Length 响应体的长度    Content-Length: 348
#define RESP_HEADER_CONTENT_LOCATION "Content-Location"  // Resp_Content_Location 请求资源可替代的备用的另一地址    Content-Location: /index.htm
#define RESP_HEADER_CONTENT_MD5 "Content-MD5"            // Resp_Content_MD5 返回资源的MD5校验值    Content-MD5: Q2hlY2sgSW50ZWdyaXR5IQ==
#define RESP_HEADER_CONTENT_RANGE "Content-Range"        // Resp_Content_Range 在整个返回体中本部分的字节位置    Content-Range: bytes 21010-47021/47022
#define RESP_HEADER_CONTENT_TYPE "Content-Type"          // Resp_Content_Type 返回内容的MIME类型    Content-Type: text/html; charset=utf-8
#define RESP_HEADER_DATE "Date"                          // Resp_Date 原始服务器消息发出的时间    Date: Tue, 15 Nov 2010 08:12:31 GMT
#define RESP_HEADER_ETAG "ETag"                          // Resp_ETag 请求变量的实体标签的当前值    ETag: “737060cd8c284d8af7ad3082f209582d”
#define RESP_HEADER_EXPIRES "Expires"                    // Resp_Expires 响应过期的日期和时间    Expires: Thu, 01 Dec 2010 16:00:00 GMT
#define RESP_HEADER_LAST_MODIFIED "Last-Modified"        // Resp_Last_Modified 请求资源的最后修改时间    Last-Modified: Tue, 15 Nov 2010 12:45:26 GMT
#define RESP_HEADER_LOCATION "Location"  // Resp_Location 用来重定向接收方到非请求URL的位置来完成请求或标识新的资源    Location: http://www.zcmhi.com/archives/94.html
#define RESP_HEADER_PRAGMA "Pragma"      // Resp_Pragma 包括实现特定的指令，它可应用到响应链上的任何接收方    Pragma: no-cache
#define RESP_HEADER_PROXY_AUTHENTICATE "Proxy-Authenticate"  // Resp_Proxy_Authenticate 它指出认证方案和可应用到代理的该URL上的参数    Proxy-Authenticate: Basic
#define RESP_HEADER_REFRESH "refresh"  // Resp_refresh 应用于重定向或一个新的资源被创造，在5秒之后重定向（由网景提出，被大部分浏览器支持）    Refresh: 5; url=http://www.zcmhi.com/archives/94.html
#define RESP_HEADER_RETRY_AFTER "Retry-After"              // Resp_Retry_After 如果实体暂时不可取，通知客户端在指定时间之后再次尝试    Retry-After: 120
#define RESP_HEADER_SERVER "Server", ;                     // Resp_Server web服务器软件名称    Server: Apache/1.3.27 (Unix) (Red-Hat/Linux)
#define RESP_HEADER_SET_COOKIE "Set-Cookie"                // Resp_Set_Cookie 设置Http Cookie    Set-Cookie: UserID=JohnDoe; Max-Age=3600; Version=1
#define RESP_HEADER_TRAILER "Trailer"                      // Resp_Trailer 指出头域在分块传输编码的尾部存在    Trailer: Max-Forwards
#define RESP_HEADER_TRANSFER_ENCODING "Transfer-Encoding"  // Resp_Transfer_Encoding 文件传输编码    Transfer-Encoding:chunked
#define RESP_HEADER_VARY "Vary"                            // Resp_Vary 告诉下游代理是使用缓存响应还是从原始服务器请求    Vary: *
#define RESP_HEADER_VIA "Via"                              // Resp_Via 告知代理客户端响应是通过哪里发送的    Via: 1.0 fred, 1.1 nowhere.com (Apache/1.1)
#define RESP_HEADER_WARNING "Warning"                      // Resp_Warning 警告实体可能存在的问题    Warning: 199 Miscellaneous warning
#define RESP_HEADER_WWW_AUTHENTICATE "WWW-Authenticate"    // Resp_WWW_Authenticate 表明客户端请求实体应该使用的授权方案    WWW-Authenticate: Basic

#define REQ_HEADER_ACCEPT "Accept"                    // Req_Accept 指定客户端能够接收的内容类型    Accept: text/plain, text/html
#define REQ_HEADER_ACCEPT_CHARSET "Accept-Charset"    // Req_Accept_Charset 浏览器可以接受的字符编码集。    Accept-Charset: iso-8859-5
#define REQ_HEADER_ACCEPT_ENCODING "Accept-Encoding"  // Req_Accept_Encoding 指定浏览器可以支持的web服务器返回内容压缩编码类型。    Accept-Encoding: compress, gzip
#define REQ_HEADER_ACCEPT_LANGUAGE "Accept-Language"  // Req_Accept_Language 浏览器可接受的语言    Accept-Language: en,zh
#define REQ_HEADER_ACCEPT_RANGES "Accept-Ranges"      // Req_Accept_Ranges 可以请求网页实体的一个或者多个子范围字段    Accept-Ranges: bytes
#define REQ_HEADER_AUTHORIZATION "Authorization"      // Req_Authorization HTTP授权的授权证书    Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
#define REQ_HEADER_CACHE_CONTROL "Cache-Control"      // Req_Cache_Control 指定请求和响应遵循的缓存机制    Cache-Control: no-cache
#define REQ_HEADER_CONNECTION "Connection"            // Req_Connection 表示是否需要持久连接。（HTTP 1.1默认进行持久连接）    Connection: close
#define REQ_HEADER_COOKIE "Cookie"  // Req_Cookie HTTP请求发送时，会把保存在该请求域名下的所有cookie值一起发送给web服务器。    Cookie: $Version=1; Skin=new;
#define REQ_HEADER_CONTENT_LENGTH "Content-Length"  // Req_Content_Length 请求的内容长度    Content-Length: 348
#define REQ_HEADER_CONTENT_TYPE "Content-Type"      // Req_Content_Type 请求的与实体对应的MIME信息    Content-Type: application/x-www-form-urlencoded
#define REQ_HEADER_DATE "Date"                      // Req_Date 请求发送的日期和时间    Date: Tue, 15 Nov 2010 08:12:31 GMT
#define REQ_HEADER_EXPECT "Expect"                  // Req_Expect 请求的特定的服务器行为    Expect: 100-continue
#define REQ_HEADER_FROM "From"                      // Req_From 发出请求的用户的Email    From: user@email.com
#define REQ_HEADER_HOST "Host"                      // Req_Host 指定请求的服务器的域名和端口号    Host: www.zcmhi.com
#define REQ_HEADER_IF_MATCH "If-Match"              // Req_If_Match 只有请求内容与实体相匹配才有效    If-Match: “737060cd8c284d8af7ad3082f209582d”
#define REQ_HEADER_IF_MODIFIED_SINCE \
    "If-Modified-Since"  // Req_If_Modified_Since 如果请求的部分在指定时间之后被修改则请求成功，未被修改则返回304代码    If-Modified-Since: Sat, 29 Oct 2010 19:43:31 GMT
#define REQ_HEADER_IF_NONE_MATCH \
    "If-None-Match"  // Req_If_None_Match 如果内容未改变返回304代码，参数为服务器先前发送的Etag，与服务器回应的Etag比较判断是否改变    If-None-Match: “737060cd8c284d8af7ad3082f209582d”
#define REQ_HEADER_IF_RANGE "If-Range"  // Req_If_Range 如果实体未改变，服务器发送客户端丢失的部分，否则发送整个实体。参数也为Etag    If-Range: “737060cd8c284d8af7ad3082f209582d”
#define REQ_HEADER_IF_UNMODIFIED_SINCE "If-Unmodified-Since"  // Req_If_Unmodified_Since 只在实体在指定时间之后未被修改才请求成功    If-Unmodified-Since: Sat, 29 Oct 2010 19:43:31 GMT
#define REQ_HEADER_MAX_FORWARDS "Max-Forwards"                // Req_Max_Forwards 限制信息通过代理和网关传送的时间    Max-Forwards: 10
#define REQ_HEADER_PRAGMA "Pragma"                            // Req_Pragma 用来包含实现特定的指令    Pragma: no-cache
#define REQ_HEADER_PROXY_AUTHORIZATION "Proxy-Authorization"  // Req_Proxy_Authorization 连接到代理的授权证书    Proxy-Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==
#define REQ_HEADER_RANGE "Range"                              // Req_Range 只请求实体的一部分，指定范围    Range: bytes=500-999
#define REQ_HEADER_REFERER "Referer"                          // Req_Referer 先前网页的地址，当前请求网页紧随其后,即来路    Referer: http://www.zcmhi.com/archives/71.html
#define REQ_HEADER_TE "TE"                                    // Req_TE 客户端愿意接受的传输编码，并通知服务器接受接受尾加头信息    TE: trailers,deflate;q=0.5
#define REQ_HEADER_UPGRADE "Upgrade"        // Req_Upgrade 向服务器指定某种传输协议以便服务器进行转换（如果支持）    Upgrade: HTTP/2.0, SHTTP/1.3, IRC/6.9, RTA/x11
#define REQ_HEADER_USER_AGENT "User-Agent"  // Req_User_Agent User-Agent的内容包含发出请求的用户信息    User-Agent: Mozilla/5.0 (Linux; X11)
#define REQ_HEADER_VIA "Via"                // Req_Via 通知中间网关或代理服务器地址，通信协议    Via: 1.0 fred, 1.1 nowhere.com (Apache/1.1)
#define REQ_HEADER_WARNING "Warning"        // Req_Warning 关于消息实体的警告信息    Warn: 199 Miscellaneous warning

enum SDKTYPE {
    OPENSSL = 0,        //openssl only
    OPENSSL_AP,         //openssl + ap crypto
    OPENSSL_JIT,        //openssl + jit
};

enum CertExistStatus {
    CSTATUS_NOKEY_NOCERT        = 0, //密钥及证书都不存在 
    CSTATUS_WITHKEY_NOCERT      = 1, //密钥存在，证书不存在 
    CSTATUS_NOKEY_WITHCERT      = 2, // 密钥不存在，证书存在 
    CSTATUS_WITHKEY_WITHCERT    = 3, // 密钥存在，证书存在 
};

enum Status {
  init = 0,     // prepare to download file.
  running = 1,  // running
  suspend = 2,  // download process suspend
  fail = 3,     // download process fail
  complete = 4  // download process complete
};

enum HttpsMethod {
    HTTPS_GET = 0,
    HTTPS_POST,
    HTTPS_PUT,
    HTTPS_DELETE,
};

struct Request {
    int method = HTTP_GET;
    std::string url;
    std::string save_file_path;
    std::map<std::string, std::string> headers;
    std::string post_data;
    std::shared_ptr<std::vector<uint8_t>> post_data_v2;

    int sdkType;
    std::string client_ap_priv_key_slot;
    std::string client_priv_key_file;
    std::string client_cert_chain;
    std::string client_key_cert_p12;
    std::string client_key_cert_p12_pass;
    std::string root_ca_cert_file;
};

struct Response {
    int code;
    int id;
    std::map<std::string, std::string> headers;
    std::string content;
    std::uint32_t rate_of_download;
    Status status_download;
};

struct DnInfo {
    std::string country;
    std::string organization;
    std::string organization_unit;
    std::string state;
    std::string common_name;
    std::string email_address;
};

struct JitInitPara {
    std::string logPath;
    std::string keyPath;
    std::string certPath;        
    std::string ca_path_name;   // root ca cert file
    std::string pfx_path_name;  // preinstall cert file
    std::string pfxPin;
    std::string keycertPin;
    std::string slot;
    std::string device_cert_port;
    std::string device_cert_file_name;
};

using RequestPtr = std::shared_ptr<Request>;
using ResponsePtr = std::shared_ptr<Response>;
using ResponseHandler = std::function<void(int id, ResponsePtr resp_ptr)>;
using DataBuffer = std::shared_ptr<std::vector<uint8_t>>;

#define make_shared_data_buffer(x) std::shared_ptr<std::vector<uint8_t>>(new std::vector<uint8_t>(x));

#define make_shared_curl_single_handle() std::shared_ptr<CURL>(curl_easy_init(), [](CURL* handle) { curl_easy_cleanup(handle); })
#define make_shared_curl_multi_handle() std::shared_ptr<CURL>(curl_multi_init(), [](CURL* handle) { curl_multi_cleanup(handle); })

#define INVALID_HTTP_ID (-1)

#ifndef UNUSED_VARIABLE
#define USED_VARIABLE(v) (void)(v)
#endif

}  // namespace https
}  // namespace v2c
}  // namespace hozon

#endif
