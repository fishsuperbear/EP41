/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: datamanager
 */

#include "include/cfg_manager.h"

#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>

namespace hozon {
namespace netaos {
namespace cfg {
using namespace hozon::netaos::per;
using namespace hozon::netaos::phm;

CfgManager* CfgManager::sinstance_ = nullptr;
std::mutex g_cfgmgr_mutex;
CfgManager* CfgManager::Instance() {
    if (nullptr == sinstance_) {
        std::lock_guard<std::mutex> lck(g_cfgmgr_mutex);
        sinstance_ = new CfgManager();
    }
    return sinstance_;
}

CfgManager::CfgManager() { cfgfilename = "config_param.json"; }

CfgManager::~CfgManager() {}

void CfgManager::Init(std::string dir_path, std::string redundant_path, uint32_t maxcom_vallimit) {
    redundant_path_ = redundant_path;
    maxcom_vallimit_ = maxcom_vallimit;
    dir_path_ = dir_path;
}
void CfgManager::DeInit() {}

bool CfgManager::ReadCfgFile(CfgServerData& datacfg) {
    StorageConfig config;
    config.redundancy_config.auto_recover = true;
    config.redundancy_config.redundant_count = 1;
    config.redundancy_config.redundant_dirpath = redundant_path_ + commondir;
    auto fs = OpenFileStorage(dir_path_ + commondir + cfgfilename, config);
    if (!fs) {
        CONFIG_LOG_ERROR << "open file storage failed.\n";
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    SharedHandle<FileStorage> fsResult = std::move(fs).Value();
    if (!fsResult) {
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        CONFIG_LOG_ERROR << "open wFile failed.\n";
        return false;
    }
    hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> rFileResult = fsResult->OpenFileReadOnly(BasicOperations::OpenMode::kIn);
    if (!rFileResult.HasValue()) {
        CONFIG_LOG_ERROR << "OpenFileReadOnly failed due to:. " << rFileResult.Error().Value();
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    UniqueHandle<ReadAccessor> rFile = std::move(rFileResult).Value();
    if (!rFile) {
        CONFIG_LOG_ERROR << "open rFile failed.\n";
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    auto res = fsResult->ReadObject(datacfg, *rFile);
    if (!res.HasValue()) {
        CONFIG_LOG_ERROR << "ReadObject failed";
        SendFault_t fault(4180, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    std::streampos rv = std::move(res).Value();
    CONFIG_LOG_INFO << "ReadObject " << rv;

    for (auto it = datacfg.cfgParamDataMap_.begin(); it != datacfg.cfgParamDataMap_.end(); ++it) {
        std::string dirfilepath;
        std::string filekey;
        std::string filepath;
        string::size_type pos = it->first.find("/");
        bool findflag = false;
        if (pos != it->first.npos) {
            findflag = true;
            dirfilepath = it->first.substr(0, pos);
            filekey = it->first.substr(pos + 1, it->first.size());
            filepath = dirfilepath + "/" + filekey + ".bin";
            config.redundancy_config.redundant_dirpath = redundant_path_ + "/" + dirfilepath;
        } else {
            dirfilepath = it->first;
            filepath = dirfilepath + ".bin";
            config.redundancy_config.redundant_dirpath = redundant_path_;
        }
        if (it->second.storageFlag == CONFIG_REF_FILE) {
            auto fs1 = OpenFileStorage(dir_path_ + filepath, config);
            if (!fs1) {
                CONFIG_LOG_ERROR << "open file storage failed.\n";
                SendFault_t fault(4170, 2, 1);
                int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                if (result < 0) {
                    CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                }
                continue;
            }
            SharedHandle<FileStorage> fsResult1 = std::move(fs1).Value();
            if (!fsResult1) {
                CONFIG_LOG_ERROR << "open wFile failed.\n";
                continue;
            }
            hozon::netaos::core::Result<UniqueHandle<ReadAccessor>> rFileResult1 = fsResult1->OpenFileReadOnly(BasicOperations::OpenMode::kBinary);
            if (!rFileResult1.HasValue()) {
                CONFIG_LOG_ERROR << "OpenFileReadWrite failed due to:. " << rFileResult1.Error().Value();
                SendFault_t fault(4170, 2, 1);
                int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                if (result < 0) {
                    CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                }
                continue;
            }
            UniqueHandle<ReadAccessor> rFile1 = std::move(rFileResult1).Value();
            if (!rFile1) {
                CONFIG_LOG_ERROR << "open rFile1 failed.\n";
                SendFault_t fault(4170, 2, 1);
                int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                if (result < 0) {
                    CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                }
                continue;
            }
            rFile1->seek(0, BasicOperations::SeekDirection::kEnd);
            uint32_t filesize = rFile1->tell();
            rFile1->seek(0, BasicOperations::SeekDirection::kBeg);
            char* readBuffer = new char[filesize];
            hozon::netaos::core::Span<char> readData(reinterpret_cast<char*>(readBuffer), filesize);
            auto readSize = rFile1->readbinary(readData);
            std::string str(readData.data(), readData.size());
            datacfg.cfgParamDataMap_[it->first].paramValue = str;
            // datacfg.cfgParamDataMap_[it->first].paramValue.clear();
            // datacfg.cfgParamDataMap_[it->first].paramValue.insert(datacfg.cfgParamDataMap_[it->first].paramValue.begin(), readData.begin(), readData.end());
            delete[] readBuffer;
            if (readSize != (int32_t)readData.size()) {
                CONFIG_LOG_ERROR << "read failed " << readSize << ":" << filesize;
                SendFault_t fault(4170, 4, 1);
                int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                if (result < 0) {
                    CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                }
                continue;
            }
        } else if (it->second.storageFlag == CONFIG_REF_KEY_VALUE) {
            if (findflag) {
                config.serialize_format = "json";
                auto result1 = OpenKeyValueStorage(dir_path_ + dirfilepath + "/" + dirfilepath + ".json", config);
                if (!result1) {
                    CONFIG_LOG_ERROR << "open key value storage failed.";
                    SendFault_t fault(4170, 3, 1);
                    int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result2 < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
                    }
                    continue;
                }
                SharedHandle<KeyValueStorage> kvs = std::move(result1).Value();
                if (!kvs) {
                    CONFIG_LOG_ERROR << "open wFile failed.";
                    SendFault_t fault(4170, 3, 1);
                    int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result2 < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
                    }
                    continue;
                }
                if (!kvs->GetValue(filekey, datacfg.cfgParamDataMap_[it->first].paramValue)) {
                    CONFIG_LOG_ERROR << "GetValue failed.";
                    SendFault_t fault(4170, 6, 1);
                    int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result2 < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
                    }
                    continue;
                }
            } else {
                datacfg.cfgParamDataMap_[it->first].storageFlag = CONFIG_KEY_VALUE;
            }
        } else {
            if (datacfg.cfgParamDataMap_[it->first].defaultparamValue.size()) {
                datacfg.cfgParamDataMap_[it->first].paramValue = datacfg.cfgParamDataMap_[it->first].defaultparamValue;
            }
        }
    }
    return true;
}

bool CfgManager::ReadKvsCfgFile(CfgServerData& datacfg, std::string paramName, uint8_t type) {
    StorageConfig config;
    config.redundancy_config.auto_recover = true;
    config.redundancy_config.redundant_count = 1;
    config.redundancy_config.redundant_dirpath = redundant_path_;
    std::string::size_type pos = paramName.find_last_of("/");
    if (pos == paramName.npos) {
        CONFIG_LOG_INFO << "not find key";
        return false;
    }
    std::string dirfilepath = paramName.substr(0, pos);
    std::string filekey = paramName.substr(pos + 1, paramName.size());
    config.redundancy_config.redundant_dirpath = redundant_path_ + "/" + dirfilepath;
    config.serialize_format = "json";
    std::string filpath = dir_path_ + dirfilepath + "/" + dirfilepath + ".json";
    struct stat buffer;
    if (stat(filpath.c_str(), &buffer) != 0) {
        CONFIG_LOG_INFO << "not find file " << filpath;
        return false;
    }
    auto result1 = OpenKeyValueStorage(dir_path_ + dirfilepath + "/" + dirfilepath + ".json", config);
    if (!result1) {
        CONFIG_LOG_ERROR << "open key value storage failed.";
        SendFault_t fault(4170, 3, 1);
        int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result2 < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
        }
        return false;
    }
    SharedHandle<KeyValueStorage> kvs = std::move(result1).Value();
    if (!kvs) {
        CONFIG_LOG_ERROR << "open wFile failed.";
        SendFault_t fault(4170, 3, 1);
        int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result2 < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
        }
        return false;
    }
    if (!kvs->GetValue(filekey, datacfg.cfgParamDataMap_[paramName].paramValue)) {
        CONFIG_LOG_ERROR << "GetValue failed.";
        SendFault_t fault(4170, 6, 1);
        int32_t result2 = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result2 < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result2;
        }
        return false;
    }
    datacfg.cfgParamDataMap_[paramName].storageFlag = CONFIG_REF_KEY_VALUE;
    datacfg.cfgParamDataMap_[paramName].perFlag = CONFIG_SYNC_PERSIST;
    datacfg.cfgParamDataMap_[paramName].dataSize = datacfg.cfgParamDataMap_[paramName].paramValue.size();
    datacfg.cfgParamDataMap_[paramName].dataType = type;
    CONFIG_LOG_INFO << "res. true";
    return true;
}

bool CfgManager::WriteCfgFile(CfgServerData& datacfg, std::string key) {
    StorageConfig config;
    config.redundancy_config.auto_recover = true;
    config.redundancy_config.redundant_count = 1;
    for (auto it = datacfg.cfgParamDataMap_.begin(); it != datacfg.cfgParamDataMap_.end(); ++it) {
        std::string dirfilepath;
        std::string filekey;
        std::string filepath;
        string::size_type pos = it->first.find("/");
        bool findflag = false;
        if (pos != it->first.npos) {
            findflag = true;
            dirfilepath = it->first.substr(0, pos);
            filekey = it->first.substr(pos + 1, it->first.size());
            filepath = dirfilepath + "/" + filekey + ".bin";
            config.redundancy_config.redundant_dirpath = redundant_path_ + "/" + dirfilepath;
        } else {
            dirfilepath = it->first;
            filepath = dirfilepath + ".bin";
            config.redundancy_config.redundant_dirpath = redundant_path_;
        }
        if (it->first == key) {
            if (datacfg.cfgParamDataMap_[key].dataSize > maxcom_vallimit_) {
                datacfg.cfgParamDataMap_[key].storageFlag = CONFIG_REF_FILE;
                auto fs = OpenFileStorage(dir_path_ + filepath, config);
                if (!fs) {
                    CONFIG_LOG_ERROR << "open file storage failed.\n";
                    SendFault_t fault(4170, 2, 1);
                    int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                    }
                    return false;
                }
                SharedHandle<FileStorage> fsResult = std::move(fs).Value();
                if (!fsResult) {
                    CONFIG_LOG_ERROR << "open wFile failed.\n";
                    SendFault_t fault(4170, 2, 1);
                    int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                    }
                    return false;
                }
                hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> wFileResult = fsResult->OpenFileReadWrite(BasicOperations::OpenMode::kOut);
                if (!wFileResult.HasValue()) {
                    CONFIG_LOG_ERROR << "OpenFileReadWrite failed due to:. " << wFileResult.Error().Value();
                    SendFault_t fault(4170, 2, 1);
                    int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                    }
                    return false;
                }
                UniqueHandle<ReadWriteAccessor> wFile = std::move(wFileResult).Value();
                if (!wFile) {
                    CONFIG_LOG_ERROR << "open wFile failed.\n";
                    SendFault_t fault(4170, 2, 1);
                    int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                    }
                    return false;
                }
                hozon::netaos::core::Span<char> writeData(const_cast<char*>(datacfg.cfgParamDataMap_[key].paramValue.c_str()), datacfg.cfgParamDataMap_[key].paramValue.size());
                auto writeSize = wFile->writebinary(writeData);
                if (writeSize != (int32_t)datacfg.cfgParamDataMap_[key].paramValue.size()) {
                    CONFIG_LOG_ERROR << "write failed.\n";
                    SendFault_t fault(4170, 5, 1);
                    int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
                    if (result < 0) {
                        CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
                    }
                    return false;
                }
                datacfg.cfgParamDataMap_[key].paramValue.clear();
                wFile->fsync();
            } else {
                if (findflag) {
                    config.serialize_format = "json";
                    datacfg.cfgParamDataMap_[key].storageFlag = CONFIG_REF_KEY_VALUE;
                    auto result = OpenKeyValueStorage(dir_path_ + dirfilepath + "/" + dirfilepath + ".json", config);
                    if (!result) {
                        CONFIG_LOG_ERROR << "open key value storage failed.";
                        SendFault_t fault(4170, 3, 1);
                        int32_t result1 = PhmClientInstance::getInstance()->ReportFault(fault);
                        if (result1 < 0) {
                            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result1;
                        }
                        return false;
                    }
                    SharedHandle<KeyValueStorage> kvs = std::move(result).Value();
                    if (!kvs) {
                        CONFIG_LOG_ERROR << "open wFile failed.";
                        SendFault_t fault(4170, 3, 1);
                        int32_t result1 = PhmClientInstance::getInstance()->ReportFault(fault);
                        if (result1 < 0) {
                            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result1;
                        }
                        return false;
                    }
                    if (!kvs->SetValue(filekey, datacfg.cfgParamDataMap_[key].paramValue)) {
                        CONFIG_LOG_ERROR << "SetValue failed.";
                        SendFault_t fault(4170, 7, 1);
                        int32_t result1 = PhmClientInstance::getInstance()->ReportFault(fault);
                        if (result1 < 0) {
                            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result1;
                        }
                        return false;
                    }
                    kvs->SyncToStorage();
                    datacfg.cfgParamDataMap_[key].paramValue.clear();
                } else {
                    datacfg.cfgParamDataMap_[key].storageFlag = CONFIG_KEY_VALUE;
                }
            }
        } else {
            if (datacfg.cfgParamDataMap_[it->first].storageFlag) {
                datacfg.cfgParamDataMap_[it->first].paramValue.clear();
            }
        }
    }

    config.redundancy_config.redundant_dirpath = redundant_path_ + commondir;
    auto fs = OpenFileStorage(dir_path_ + commondir + cfgfilename, config);
    if (!fs) {
        CONFIG_LOG_ERROR << "open file storage failed.\n";
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    SharedHandle<FileStorage> fsResult = std::move(fs).Value();
    if (!fsResult) {
        CONFIG_LOG_ERROR << "open wFile failed.\n";
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    hozon::netaos::core::Result<UniqueHandle<ReadWriteAccessor>> rFileResult = fsResult->OpenFileReadWrite(BasicOperations::OpenMode::kOut);
    if (!rFileResult.HasValue()) {
        CONFIG_LOG_ERROR << "OpenFileReadWrite failed due to:. " << rFileResult.Error().Value();
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    UniqueHandle<ReadWriteAccessor> rFile = std::move(rFileResult).Value();
    if (!rFile) {
        CONFIG_LOG_ERROR << "open rFile failed.\n";
        SendFault_t fault(4170, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    auto res = fsResult->WriteObject(datacfg, *rFile, true);
    if (!res.HasValue()) {
        CONFIG_LOG_ERROR << "WriteObject failed";
        SendFault_t fault(4180, 2, 1);
        int32_t result = PhmClientInstance::getInstance()->ReportFault(fault);
        if (result < 0) {
            CONFIG_LOG_WARN << "ReportFault failed. failedCode: " << result;
        }
        return false;
    }
    std::streampos rv = std::move(res).Value();
    CONFIG_LOG_INFO << "WriteObject " << rv;

    return true;
}
}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
