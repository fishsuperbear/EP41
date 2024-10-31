#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <algorithm>
#include "json/json.h"

namespace hozon {
namespace netaos {
namespace update {

class OtaResultStore {
public:

    static OtaResultStore* Instance();

    void Init();
    void Deinit();
    bool ResetResult();
    bool UpdateResult(const std::string& key, const std::string& value);
    bool GetResultByKey(const std::string& key, std::string& value);
    std::string To_string();

private:
    OtaResultStore();
    ~OtaResultStore();
    OtaResultStore(const OtaResultStore &);
    OtaResultStore & operator = (const OtaResultStore &);
private:
    void writeToJsonFile(const Json::Value &root);
    Json::Value readFromJsonFile();
    bool isValidKey(const std::string &key);
    bool isValidValue(const std::string &value);

    static std::mutex m_mtx;
    static OtaResultStore* m_pInstance;

    const std::vector<std::string> keys_;
    const std::vector<std::string> validValues_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
