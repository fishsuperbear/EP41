#include "DoSomeIPFuntions.h"

using namespace hozon::netaos::diag;

void PrintRequest(const v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage& _req) {
    DS_INFO << "v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage:";
    DS_INFO << "udsSa: " << UINT16_TO_STRING(_req.getUdsSa());
    DS_INFO << "udsTa: " << UINT16_TO_STRING(_req.getUdsTa());
    DS_INFO << "taType: " << _req.getTaType().toString();
    std::vector<uint8_t> tmp = _req.getUdsData();
    DS_INFO << "udsData size: " << tmp.size();
    DS_INFO << "udsData: " << UINT8_VEC_TO_STRING(tmp);
}

void PrintRequest(const hozon::netaos::diag::DoSomeIPReqUdsMessage& req) {
    DS_INFO << "hozon::netaos::diag::DoSomeIPReqUdsMessage:";
    DS_INFO << "  udsSa: " << UINT16_TO_STRING(req.udsSa);
    DS_INFO << "  udsTa: " << UINT16_TO_STRING(req.udsTa);
    DS_INFO << "  taType: " << GetAddressTypeString(req.taType);
    DS_INFO << "udsData size: " << req.udsData.size();
    DS_INFO << "udsData: " << UINT8_VEC_TO_STRING(req.udsData);
}

void PrintResponse(const v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& resp) {
    DS_INFO << "v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage:";
    DS_INFO << "udsSa: " << UINT16_TO_STRING(resp.getUdsSa());
    DS_INFO << "udsTa: " << UINT16_TO_STRING(resp.getUdsTa());
    DS_INFO << "result: " << resp.getResult();
    DS_INFO << "taType: " << resp.getTaType().toString();
    std::vector<uint8_t> tmp = resp.getUdsData();
    DS_INFO << "udsData size: " << tmp.size();
    DS_INFO << "udsData: " << UINT8_VEC_TO_STRING(tmp);
}

void PrintResponse(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp) {
    DS_INFO << "hozon::netaos::diag::DoSomeIPRespUdsMessage:";
    DS_INFO << "udsSa: " << UINT16_TO_STRING(resp.udsSa);
    DS_INFO << "udsTa: " << UINT16_TO_STRING(resp.udsTa);
    DS_INFO << "result: " << resp.result;
    DS_INFO << "taType: " << GetAddressTypeString(resp.taType);
    DS_INFO << "udsData size: " << resp.udsData.size();
    DS_INFO << "udsData: " << UINT8_VEC_TO_STRING(resp.udsData);
}

bool ConvertStruct(const v1::commonapi::DoSomeIP::DoSomeIPReqUdsMessage& _req, hozon::netaos::diag::DoSomeIPReqUdsMessage& req) {
    DS_INFO << "ConvertStruct Req: ";
    req.udsSa = _req.getUdsSa();
    req.udsTa = _req.getUdsTa();

    if (_req.getTaType() == 0) {
        req.taType = hozon::netaos::diag::TargetAddressType::kPhysical;
    } else if (_req.getTaType() == 1) {
        req.taType = hozon::netaos::diag::TargetAddressType::kFunctional;
    } else {
        //do nothing
    }
    req.udsData = _req.getUdsData();
    DS_INFO << "After ConvertStruct Req: ";
    PrintRequest(req);
    return true;
}

bool ConvertStruct(const hozon::netaos::diag::DoSomeIPRespUdsMessage& resp, v1::commonapi::DoSomeIP::DoSomeIPRespUdsMessage& _resp) {
    DS_INFO << "ConvertStruct Resp: ";
    _resp.setUdsSa(resp.udsSa);
    _resp.setUdsTa(resp.udsTa);
    _resp.setResult(resp.result);
    v1::commonapi::DoSomeIP::TargetAddressType type = v1::commonapi::DoSomeIP::TargetAddressType::kPhysical;
    if (resp.taType == hozon::netaos::diag::TargetAddressType::kPhysical) {
        _resp.setTaType(type);
    } else if (resp.taType == hozon::netaos::diag::TargetAddressType::kFunctional) {
        type = v1::commonapi::DoSomeIP::TargetAddressType::kFunctional;
        _resp.setTaType(type);
    } else {
        //do nothing
    }
    _resp.setUdsData(resp.udsData);
    DS_INFO << "After ConvertStruct Resp: ";
    PrintResponse(_resp);
    return true;
}

std::string GetAddressTypeString(const hozon::netaos::diag::TargetAddressType& type)
{
    switch (type)
    {
        case TargetAddressType::kPhysical:
            return "kPhysical";
        case TargetAddressType::kFunctional:
            return "kFunctional";
        default:
            return "NULL";
    }
}

