/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: devm version check
 */
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <map>

#include "json/json.h"
#include "devm_data_define.h"
#include "device_info.h"
#include "devm_server_logger.h"
#include "devm_version_check.h"
#include "zmq_ipc/proto/devm_tool.pb.h"
#include "function_statistics.h"
#include "cfg_data.hpp"


namespace hozon {
namespace netaos {
namespace devm_server {

DevmVerCheck::DevmVerCheck()
    : cfg_mgr_(ConfigParam::Instance())
{

}

DevmVerCheck::~DevmVerCheck() {
}

bool
DevmVerCheck::ReadMajorAndSWTVersionFromFile(std::string& major_v, std::string& swt_v) {
    std::string version{};
    std::string filename = "/app/version.json";
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        DEVM_LOG_ERROR << "parseJson error, message: " << errs;
        return false;
    }
    ifs.close();
    std::string str = root["HZ"].asString();

    // get string "03.01.05" from "EP41_ORIN_HZdev_03.01.05_0704_1017_20231020"
    size_t start = 0;
    size_t end;
    for (int i = 0; i <= 3; i++) {
        end = str.find("_", start);
        if (end == std::string::npos) {
            return false;
        }
        end++;
	    if (i < 3) {
            start = end;
        }
    }
    major_v = str.substr(start, end - start - 1);
    swt_v = root["SW"].asString();

    return true;
}

std::string
DevmVerCheck::ReadSocVersionFromFile() {
    std::string version{};
    std::string filename = "/app/version.json";
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        DEVM_LOG_ERROR << "Failed to open file: " << filename;
        return version;
    }

    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        DEVM_LOG_ERROR << "parseJson error, message: " << errs;
        return version;
    }
    ifs.close();
    version = root["app_version"].asString();

    return version;
}

std::string
DevmVerCheck::ReadDsvVersionFromFile(){
    std::string version{};
    std::string filePath = "/etc/version";
    std::ifstream file(filePath);

    if (!file.is_open()) {
        DEVM_LOG_ERROR << "Failed to open file: " << filePath;
        return version;
    }

    std::string line;
    while (std::getline(file, line)) {
        version += line;
    }

    // 删除末尾的换行符
    size_t pos = version.find('\n');
    if (pos != std::string::npos) {
        version.erase(pos);
    }

    return version;
}

bool
DevmVerCheck::ReadDidsFromFile(std::map<std::string, std::string>& dids_value) {
    std::string filename = "/svp_data/EOL/InfoInjectAndCloudUp.json";
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        DEVM_LOG_ERROR << "Failed to open json file: " << filename;
        return false;
    }

    Json::CharReaderBuilder reader;
    Json::Value root;
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(reader, ifs, &root, &errs);
    if (!res || !errs.empty()) {
        DEVM_LOG_ERROR << "parseJson error, message: " << errs;
        return false;
    }
    ifs.close();

    std::string
    value = root["VehicleManufacturerSparePartNumber"].asString();
    dids_value.insert(std::make_pair(PART_NUMBER_DIDS, value));
    value = root["SystemSupplierIdentifierDataIdentifier"].asString();
    dids_value.insert(std::make_pair(SYS_SUPPLIER_ID_DIDS, value));
    value = root["ECU Manufacture Date"].asString();
    dids_value.insert(std::make_pair(ECU_MANUFACT_DATA_DIDS, value));
    value = root["ECUSerialNumberDataIdentifier"].asString();
    dids_value.insert(std::make_pair(ECU_SERIAL_NUMBER_DIDS, value));
    value = root["Vehicle Manufacturer ECU Hardware Number"].asString();
    dids_value.insert(std::make_pair(ECU_HARD_NUMBER_DIDS, value));

    return true;
}

bool 
DevmVerCheck::ReadDidsFromCfg(std::string& value, const std::string& dids)
{
    DEVM_LOG_INFO << "DevmVerCheck::ReadDidsFromCfg dids " << dids;

    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmVerCheck::ReadDidsFromCfg cfg_mgr_ is nullptr.";
        return false;
    }

    std::string data{};
    CfgResultCode res = cfg_mgr_->GetParam<std::string>(dids, data);
    if (res != CONFIG_OK && dids.find("DYNAMIC") == std::string::npos) {
        data = CfgValueInfo::getInstance()->GetCfgValueFromFile("/cfg/dids/dids.json", dids);
    }

    if (0 == data.size()) {
        DEVM_LOG_WARN << "DevmVerCheck::ReadDidsFromCfg get data is empty.";
        return false;
    }

    value = data;
    return true;
}

bool 
DevmVerCheck::WriteDidsToCfg(const std::string& value, const std::string& dids)
{
    DEVM_LOG_INFO << "DevmVerCheck::WriteDidsToCfg dids " << dids;

    if (nullptr == cfg_mgr_) {
        DEVM_LOG_ERROR << "DevmVerCheck::WriteDidsToCfg cfg_mgr_ is nullptr.";
        return false;
    }

    // TODO 校验格式

    cfg_mgr_->SetParam<std::string>(dids, value, ConfigPersistType::CONFIG_SYNC_PERSIST);
    return true;
}

std::string
DevmVerCheck::GetUpgradeStatus()
{
    client_ = std::make_shared<hozon::netaos::zmqipc::ZmqIpcClient>();//放到构造里面会报coredump，zmq库的问题
    client_->Init("tcp://localhost:11130");
    hozon::netaos::zmqipc::UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        DEVM_LOG_ERROR << "status request ret err.";
        client_->Deinit();
        return "";
    }

    hozon::netaos::zmqipc::UpgradeStatusResp status{};
    status.ParseFromString(reply);
    std::cout << "status: " << status.update_status() << std::endl;
    if (status.error_code() > 0) {
        DEVM_LOG_WARN << "Error: " << status.error_code() << ", " << status.error_msg();
        client_->Deinit();
        return "";
    }
    client_->Deinit();
    return status.update_status();
}

int32_t
DevmVerCheck::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
{
    uint8_t left = 0;
    uint8_t right = 0;

    for (uint32_t index = 0; index < bcd.size(); index++) {
        right = bcd[index];
        if (right >= '0' && right <= '9') {
            right -= '0';
        }
        else if (right >= 'A' && right <= 'F') {
            right -= 'A' - 10;
        }
        else if (right >= 'a' && right <= 'f') {
            right -= 'a' - 10;
        }
        else {
            break;
        }
        
        if (index % 2 == 1) {
            data.push_back(left << 4 | right);
        }
        else {
            left = right;
        }
    }

    return data.size();
}

void
DevmVerCheck::Run() {
/**
 * 需求：
 * 1、启动30s开始分别读取soc, mcu, dsv版本，与cfg中存储的比对，不一致写入cfg（升级模式不写入），之后每间隔30s检测一次
 * 2、启动30s读取/app/version.json，解析出大版本，与cfg中大版本比对，不一致写入cfg（升级模式不写入）
 * 3、启动30s后读取/svp_data/EOL/InfoInjectAndCloudUp.json 内容并解析，与cfg中存储的比对，不一致写入cfg 
{
   "SystemSupplierIdentifierDataIdentifier": "9DS", //dids/F18A
   "VehicleManufacturerSparePartNumber": "C40-3608110", //dids/F187
   "ECU Manufacture Date": "20180601", //dids/F18B
   "ECUSerialNumberDataIdentifier": "AEP402022B228A0001", //dids/F18C
   "Vehicle Manufacturer ECU Hardware Number": "H1.10", //dids/F191
   "supplierName": "HUIZHOU DESAY SV AUTOMOTIVECO.,LTD.",
   "adcsSoftwareVer": "03.01.00",
   "adcsType": "ADCS-OrinX"
}
*/

    //sleep 30s
    int32_t i;
    for (i = 0; i < 30 && !stop_flag_; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (i < 30) {
        return ;
    }
    DEVM_LOG_INFO << "thread version check start.";

    //read major version
    std::string major_version_file;
    std::string swt_version_file;
    ReadMajorAndSWTVersionFromFile(major_version_file, swt_version_file);
    std::string major_version_cfg;
    ReadDidsFromCfg(major_version_cfg, MAJOR_VERSION_DIDS);
    DEVM_LOG_DEBUG << "major_version_file: " << major_version_file << ", major_version_cfg: " << major_version_cfg;
    if (major_version_file.length() > 0
        && major_version_file != major_version_cfg) {
        WriteDidsToCfg(major_version_file, MAJOR_VERSION_DIDS);
        WriteDidsToCfg(major_version_file, MAJOR_VERSION_DIDS_F1C0);
        WriteDidsToCfg(major_version_file, MAJOR_VERSION_DIDS_F188);
        DeviceInfomation::getInstance()->SetMajorVersion(major_version_file);
    }


    //read /svp_data/EOL/InfoInjectAndCloudUp.json
    std::string dids_value_cfg;
    std::map<std::string, std::string> map_dids_value;
    ReadDidsFromFile(map_dids_value);
    for (const auto& pair : map_dids_value) {
        dids_value_cfg.clear();
        ReadDidsFromCfg(dids_value_cfg, pair.first);
        if (pair.first == ECU_MANUFACT_DATA_DIDS) {
            std::string str_value = pair.second;
            for (size_t i = 2; i < str_value.length(); i += 3) {
                str_value.insert(str_value.begin() + i, ' ');
            }
            map_dids_value[ECU_MANUFACT_DATA_DIDS] = str_value;
        }
        DEVM_LOG_DEBUG << "dids: " << pair.first << " file value: " << pair.second << ", cfg value: " << dids_value_cfg;
        if (pair.second.length() > 0
            && pair.second != dids_value_cfg) {
            WriteDidsToCfg(pair.second, pair.first);
        }
    }

    soc_version_dyna_.clear();
    ReadDidsFromCfg(soc_version_dyna_, SOC_VERSION_DYNAMIC);
    dsv_version_dyna_.clear();
    ReadDidsFromCfg(dsv_version_dyna_, DSV_VERSION_DYNAMIC);

    std::string version_file;
    while (!stop_flag_) {
        //soc version 
        version_file.clear();
        version_file = ReadSocVersionFromFile();
        DEVM_LOG_DEBUG << "soc version in file: " << version_file << ", cfg: " << soc_version_dyna_;
        DeviceInfomation::getInstance()->SetSocVersion(version_file);
        if (soc_version_dyna_ != version_file) {
            DEVM_LOG_INFO << "write cfg dynamic soc_version: " << version_file;
            cfg_mgr_->SetParam<std::string>(SOC_VERSION_DYNAMIC, version_file, ConfigPersistType::CONFIG_NO_PERSIST);
            soc_version_dyna_ = version_file;
        }

        //dsv version
        version_file.clear();
        version_file = ReadDsvVersionFromFile();
        DEVM_LOG_DEBUG << "dsv version in file: " << version_file << ", cfg: " << dsv_version_dyna_;
        DeviceInfomation::getInstance()->SetDsvVersion(version_file);
        if (dsv_version_dyna_ != version_file) {
            DEVM_LOG_INFO << "write cfg dynamic dsv_version: " << version_file;
            cfg_mgr_->SetParam<std::string>(DSV_VERSION_DYNAMIC, version_file, ConfigPersistType::CONFIG_NO_PERSIST);
            dsv_version_dyna_ = version_file;
        }

        //sleep 30s
        int32_t i;
        for (i = 0; i < 30 && !stop_flag_; i++) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (i < 30) {
            break;
        }
    }
    DEVM_LOG_INFO << "version check break.";
}
void
DevmVerCheck::SetStopFlag() {
    stop_flag_ = true;
}

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon
/* EOF */
