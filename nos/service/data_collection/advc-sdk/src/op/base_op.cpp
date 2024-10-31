#include "op/base_op.h"

#include "advc_sys_config.h"
#include "util/auth_tool.h"
#include "util/codec_util.h"
#include "util/http_sender.h"

namespace advc {

SignMetadata BaseOp::GetMetaData() {
    SignMetadata meta;
    meta.date = AuthTool::getV4Date();
    meta.service = kServiceName;
    meta.region = kDefaultRegion;
    meta.algorithm = "HMAC-SHA256";
    meta.credentialScope = meta.date + "/" + meta.region + "/" + meta.service + "/request";
    return meta;
}

void BaseOp::DecodeUploadToken(const std::string &in, UploadTokenStruct &uploadTokenStruct) {
    const std::string &json = in;
    Poco::JSON::Parser parser;
    Poco::Dynamic::Var result;
    try {
        result = parser.parse(json);
    } catch (const std::exception &ex) {
        SDK_LOG_ERR("parser json failed %s", std::string(ex.what()).c_str());
    }
    Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
    std::string val;
    Poco::DynamicStruct ds = *object;
    uploadTokenStruct.accessKeyId = ds["AccessKeyId"].toString();
    uploadTokenStruct.secretAccessKey = ds["SecretAccessKey"].toString();
    uploadTokenStruct.stsToken = ds["StsToken"].toString();
    if (!ds["FileName"].isEmpty()) {
        uploadTokenStruct.fileName = ds["FileName"].toString();
    }
    if (!ds["TokenQuota"].isEmpty()) {
        uploadTokenStruct.TokenQuota = std::atoi(ds["TokenQuota"].toString().c_str());
    }
    if (!ds["VehicleId"].isEmpty()) {
        uploadTokenStruct.vehicleId = ds["VehicleId"].toString();
    }
    uploadTokenStruct.customVehicleId = ds["CustomVehicleId"].toString();
    uploadTokenStruct.currentTime = ds["CurrentTime"].toString();
    uploadTokenStruct.expiredTime = ds["ExpiredTime"].toString();
}

std::string
BaseOp::SigningKeyV4(std::string secretKey, std::string &date, std::string region, std::string service) {
    std::string (*sk)(const std::string &, const std::string &);
    sk = AuthTool::ContentSha256;
    std::string req = "request";
    return sk(sk(sk(sk(secretKey, date), region), service), req);
}

std::string BaseOp::SignatureV4(std::string &signingKey, const std::string &stringToSign) {
    std::string sign = AuthTool::ContentSha256Hex(signingKey, stringToSign);
    return sign;
}

std::string BaseOp::BuildAuthHeaderV4(std::string accessKeyId, std::string &signature, SignMetadata &meta) {
    auto credential = accessKeyId + "/" + meta.credentialScope;
    std::string hearders = "content-type;host;x-content-sha256;x-date;x-security-token";
    return meta.algorithm + " Credential=" + credential + ", SignedHeaders=" + hearders + ", Signature=" +
           signature;
}

std::string BaseOp::SignRequestStr(UploadTokenStruct &uploadToken, std::map<std::string, std::string> &req_params,
                                   std::map<std::string, std::string> &req_headers, std::string httpMethod) {
    // 1.
    auto m = GetMetaData();
    std::string form_date = AuthTool::getV4Time();
    std::string headerReq = "content-type" + (std::string) ":" + req_headers.at("Content-Type") + "\n";
    headerReq += "host" + (std::string) ":" + req_headers.at("Host") + "\n";
    headerReq += "x-content-sha256" + (std::string) ":" + req_headers.at("X-Content-Sha256") + "\n";
    headerReq += "x-date" + (std::string) ":" + req_headers.at("X-Date") + "\n";
    headerReq += "x-security-token" + (std::string) ":" + req_headers.at("X-Security-Token") + "\n";

    std::string pramReq;
    pramReq += httpMethod + "\n/\n";
    for (const auto &i : req_params) {
        pramReq += i.first;
        pramReq += "=";
        pramReq += i.second;
        pramReq += "&";
    }
    pramReq[pramReq.length() - 1] = '\n';

    pramReq += headerReq + "\n" + "content-type;host;x-content-sha256;x-date;x-security-token\n" +
               req_headers.at("X-Content-Sha256");

    std::string hashedCanonReq = AuthTool::ToSha256Hex(pramReq);

    // 2.
    std::string stringToSign = "HMAC-SHA256\n";
    stringToSign += req_headers.at("X-Date") + "\n";
    stringToSign += m.credentialScope + "\n" + hashedCanonReq;

    // 3.
    std::string signDate = req_headers.at("X-Date").substr(0, 8);
    std::string signingKey = SigningKeyV4(uploadToken.secretAccessKey, signDate, kDefaultRegion, kServiceName);
    std::string SignatureStr = SignatureV4(signingKey, stringToSign);

    return BuildAuthHeaderV4(uploadToken.accessKeyId, SignatureStr, m);
}

AdvcResult BaseOp::commonRequest(UploadTokenStruct &uploadToken, std::map<std::string, std::string> _req_params,
                                 const std::string &body, const std::string httpMethod) {
    // add headers.
    std::map<std::string, std::string> req_headers;

    req_headers.insert(std::pair<std::string, std::string>("X-Security-Token", uploadToken.stsToken));
    req_headers.insert(std::pair<std::string, std::string>("Accept", "application/json"));
    req_headers.insert(std::pair<std::string, std::string>("User-Agent", "volc-sdk-cpp"));
    req_headers.insert(std::pair<std::string, std::string>("Content-Type", "application/json"));
    req_headers.insert(std::pair<std::string, std::string>("X-Date", AuthTool::getV4Time()));
    req_headers.insert(std::pair<std::string, std::string>("Host", kApiEndPoint));
    req_headers.insert(std::pair<std::string, std::string>("X-Content-Sha256", AuthTool::ToSha256Hex(body)));
    std::string authorization = SignRequestStr(uploadToken, _req_params, req_headers, httpMethod);
    req_headers.insert(std::pair<std::string, std::string>("Authorization", authorization));
    std::map<std::string, std::string> resp_headers;
    std::string resp_body;
    AdvcResult result{};
    std::string dest_url = kApiScheme + "://" + kApiEndPoint;
    std::string err_msg = "";
    // do
    int http_code = -1;

    for (int i = 0; i <= AdvcSysConfig::GetControlApiRetryTime(); i++) {
        http_code = HttpSender::SendRequest(httpMethod, dest_url, _req_params, req_headers, body,
                                            AdvcSysConfig::GetConnTimeoutInms(),
                                            AdvcSysConfig::GetRecvTimeoutInms(),
                                            AdvcSysConfig::GetSendTimeoutInms(),
                                            &resp_headers,
                                            &resp_body, &err_msg);
        if (http_code <= 499) {
            break;
        }
    }
    AdvcResult ret;
    if (resp_body.find("ResponseMetadata") != -1) {
        ret = AdvcResult::DecodeAdvcResult(resp_body);
    }
    if (http_code != 200) {
        ret.SetSuccess(false);
    }
    return ret;
}

}  // namespace advc
