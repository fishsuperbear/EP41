

#ifndef ADVC_RESULT_H
#define ADVC_RESULT_H
#pragma once

#include <map>
#include <string>

#include "advc_defines.h"
#include "advc_model.h"

namespace advc {

class AdvcResult {
   public:
    AdvcResult() : responseMetadata(), result() {}

    ~AdvcResult() = default;

    ResponseMetadata responseMetadata;
    std::string result;
    bool success;

    bool IsSuccess() {
        return success;
    }

    void SetSuccess(bool b) {
        this->success = b;
    }

    static AdvcResult DecodeAdvcResult(const std::string &in) {
        const std::string &json = in;
        Poco::JSON::Parser parser;
        Poco::Dynamic::Var result = parser.parse(json);
        AdvcResult advcResult;
        Poco::JSON::Object::Ptr object = result.extract<Poco::JSON::Object::Ptr>();
        Poco::DynamicStruct ds = *object;
        if (!ds["Result"].isEmpty()) {
            advcResult.result = ds["Result"].toString();
        }
        advcResult.responseMetadata.requestId = ds["ResponseMetadata"]["RequestId"].toString();
        advcResult.responseMetadata.service = ds["ResponseMetadata"]["Service"].toString();
        advcResult.responseMetadata.region = ds["ResponseMetadata"]["Region"].toString();
        advcResult.responseMetadata.action = ds["ResponseMetadata"]["Action"].toString();
        advcResult.responseMetadata.version = ds["ResponseMetadata"]["Version"].toString();
        if (!ds["ResponseMetadata"]["Error"].isEmpty()) {
            if (!ds["ResponseMetadata"]["Error"]["CodeN"].isEmpty() && ds["ResponseMetadata"]["Error"]["CodeN"].isInteger()) {
                std::string codeNStr = ds["ResponseMetadata"]["Error"]["CodeN"].toString();
                advcResult.responseMetadata.error.codeN = std::stoi(codeNStr);
            }
            if (!ds["ResponseMetadata"]["Error"]["Code"].isEmpty()) {
                advcResult.responseMetadata.error.code = ds["ResponseMetadata"]["Error"]["Code"].toString();
            }
            if (!ds["ResponseMetadata"]["Error"]["Message"].isEmpty()) {
                advcResult.responseMetadata.error.message = ds["ResponseMetadata"]["Error"]["Message"].toString();
            }
            advcResult.success = false;
        } else {
            advcResult.success = true;
        }
        return advcResult;
    }
};

}  // namespace advc
#endif  // ADVC_RESULT_H