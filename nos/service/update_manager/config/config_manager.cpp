#include "update_manager/config/config_manager.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/file_to_bin/file_to_bin.h"
#include "update_manager/common/common_operation.h"
#include "json/json.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/config/sensor_entity_manager.h"

#include <unistd.h>
#include <time.h>
#include <fstream>
#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/evp.h>

namespace hozon {
namespace netaos {
namespace update {

#define SENSOR_MANIFEST_NAME ("sensor_manifest.xml")
#define SENSOR_CONF_PATH (UpdateSettings::Instance().PathForUnzip() + "soc/")
#define SENSOR_CONF_FILE_PATH (UpdateSettings::Instance().PathForUnzip() + "soc/sensor/conf/")

#define SOC_MANIFEST_NAME ("manifest.xml")
#define SOC_CONF_FILE_PATH (UpdateSettings::Instance().PathForUnzip())

#define FILE_LIST_FILE ("file_list.json")

#define BIN_FILES_PATH (UpdateSettings::Instance().PathForBinFiles())
#define PAKAGE_UNZIP_PATH (UpdateSettings::Instance().PathForUnzip())

#define SENSOR_MOUNT_TARGET ("soc/sensor/")
#define AES_SENSOR_IMG_PATH ("sorssen/sorssen.ext4.enc")
#define SENSOR_IMG_PATH ("sorssen/sorssen.ext4")
#define AES_SENSOR_IMG_PATH_NEW ("soc/sorssen/sorssen.ext4.enc")
#define SENSOR_IMG_PATH_NEW ("soc/sorssen/sorssen.ext4")

ConfigManager::ConfigManager()
{
}

int32_t ConfigManager::Init()
{
    UM_INFO << "ConfigManager::Init.";
    UM_INFO << "ConfigManager::Init Done.";
    return 0;
}

int32_t ConfigManager::Start()
{
    UM_INFO << "ConfigManager::Start.";
    ParseSensorManifest();
    ParseSocManifest();
    ParseUpdateFileList();
    UM_INFO << "ConfigManager::Start Done.";
    return 0;
}

int32_t ConfigManager::Stop()
{
    UM_INFO << "ConfigManager::Stop.";
    UM_INFO << "ConfigManager::Stop.";
    return 0;
}

int32_t ConfigManager::Deinit()
{
    UM_INFO << "ConfigManager::Deinit.";
    UM_INFO << "ConfigManager::Deinit Done.";
    return 0;
}

bool ConfigManager::IsSensorUpdate()
{
    return sensor_manifest_.sensors.size() != 0;
}

bool ConfigManager::IsSocUpdate()
{
    return soc_manifest_.socs.size() != 0;
}

bool 
ConfigManager::ClearSensorUpdate(){
    sensor_manifest_.sensors.clear();
    return true;
}
bool 
ConfigManager::ClearSocUpdate(){
    soc_manifest_.socs.clear();
    return true;
}

bool ConfigManager::ParseAllConfig()
{
    UPDATE_LOG_D("ParseAllConfig");
    auto p1 = ParseSensorManifest();
    auto p2 = ParseSocManifest();
    auto p3 = ParseUpdateFileList();
    // 任意文件解析出错
    if (p1 == -1 || p2 == -1 || p3 ==-1) {
        return false;
    }
    // soc sensor xml 文件都不存在，认为解析出错
    if (p1 == -2 && p2 == -2) {
        return false;
    }
    UPDATE_LOG_D("ParseAllConfig success!");
    return true;
}

int32_t
ConfigManager::ParseUpdateFileList()
{
    UPDATE_LOG_D("ParseUpdateFileList");
    version_info_.majorVersion = "";

    std::string fileListFile = PAKAGE_UNZIP_PATH + FILE_LIST_FILE;
    UPDATE_LOG_D("fileListFile: %s", fileListFile.c_str());

    if (!PathExists(fileListFile)){
        UPDATE_LOG_W("file_list.json not Exists");
        return -2;
    }

    Json::Value rootReder;
    Json::CharReaderBuilder readBuilder;
    std::ifstream ifs(fileListFile);
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(readBuilder, ifs, &rootReder, &errs);
    if(!res)
    {
        ifs.close();
        UPDATE_LOG_E("Json parse error.");
        return -1;
    }
    const Json::Value version = rootReder["version"];
    if (!version.isNull()) {
        version_info_.majorVersion = version["version"].asString();
    } else {
        UPDATE_LOG_E("JSON key 'version' not found.");
        return -1;
    }
    ifs.close();

    return 0;
}

int32_t
ConfigManager::ParseSensorManifest()
{
    UPDATE_LOG_D("ParseSensorManifest");
    //解析sensor_manifest.xml放到sensor_manifest_
    sensor_manifest_.version = "";
    sensor_manifest_.processProportion = "";
    sensor_manifest_.sensors.clear();

    std::string sensorManifestFile = SENSOR_CONF_FILE_PATH + SENSOR_MANIFEST_NAME;
    UPDATE_LOG_D("sensorManifestFile: %s", sensorManifestFile.c_str());

    if (!PathExists(sensorManifestFile)){
        UPDATE_LOG_W("sensor manifest xml not Exists");
        return -2;
    }
    HzXml hzxmlSensor;
    std::string strValue;
    if (!hzxmlSensor.ParseXmlFile((sensorManifestFile).c_str())) {
        UPDATE_LOG_E("ParseXmlFile sensor_manifest.xml failed");
        return -1;
    }

    XMLElement* root = hzxmlSensor.GetRootElement();
    hzxmlSensor.GetChildText(root, "Version", sensor_manifest_.version);
    hzxmlSensor.GetChildText(root, "ProcessProportion", sensor_manifest_.processProportion);
    XMLElement* sensorx = hzxmlSensor.GetChildElement(root, "Sensor");

    FileToBin fileToBinHdl;
    while(sensorx)
    {
        Sensor_t     sensor;
        sensor.sensorInfo.name = "";
        sensor.sensorInfo.updateType = 0;
        sensor.sensorInfo.canidTx = 0x0000;
        sensor.sensorInfo.canidRx = 0x0000;
        sensor.sensorInfo.ip = "";
        sensor.sensorInfo.logicalAddr = 0x0000;
        sensor.sensorInfo.functionAddr = 0x0000;
        sensor.sensorInfo.havaFileSystem = false;
        sensor.sensorInfo.updateSequence = 0;
        sensor.sensorInfo.progressWeight = 0;

        hzxmlSensor.GetChildText(sensorx, "Name", sensor.sensorInfo.name);
        hzxmlSensor.GetChildText(sensorx, "UpdateType", strValue);
        sensor.sensorInfo.updateType = "docan" == strValue ? 1
                              : "doip"  == strValue ? 2
                              : "interface" == strValue ? 3
                              : "someip" == strValue ? 4
                              : "doip2docan" == strValue ? 5
                              : 0;
        if(hzxmlSensor.GetChildText(sensorx, "CanidTx", strValue)) {
            sensor.sensorInfo.canidTx = std::stoi(strValue, 0, 16);
        }
        if (hzxmlSensor.GetChildText(sensorx, "CanidRx", strValue)) {
            sensor.sensorInfo.canidRx = std::stoi(strValue, 0, 16);
        }
        hzxmlSensor.GetChildText(sensorx, "Ip", sensor.sensorInfo.ip);
        if (hzxmlSensor.GetChildText(sensorx, "LogicalAddr", strValue)) {
            sensor.sensorInfo.logicalAddr = std::stoi(strValue, 0, 16);
        }
        if (hzxmlSensor.GetChildText(sensorx, "FunctionAddr", strValue)) {
            sensor.sensorInfo.functionAddr = std::stoi(strValue, 0, 16);
        }
        hzxmlSensor.GetChildText(sensorx, "HavaFileSystem", strValue);
        sensor.sensorInfo.havaFileSystem = "True" == strValue ? true
                                         : "true" == strValue ? true
                                         : false;
        hzxmlSensor.GetChildText(sensorx, "SameVersionCheck", strValue);
        sensor.sensorInfo.sameVersionCheck = "False" == strValue ? false
                                           : "false" == strValue ? false
                                           : true;
        if (hzxmlSensor.GetChildText(sensorx, "UpdateSequence", strValue)) {
            sensor.sensorInfo.updateSequence = std::stoi(strValue);
        }
        if (hzxmlSensor.GetChildText(sensorx, "ProgressWeight", strValue)) {
            sensor.sensorInfo.progressWeight = strtof(strValue.c_str(), nullptr);
        }
        
        XMLElement* sensorEntitys = hzxmlSensor.GetChildElement(sensorx, "Entity");
        std::string randomProcessFile{};
        while (sensorEntitys) {
            SensorEntityInfo entityInfo{};
            entityInfo.partNumber = "";
            entityInfo.supplierCode = "";
            entityInfo.targetVersion = "";
            entityInfo.flashDriverFirmwareName = "";
            entityInfo.appFirmwareName = "";
            entityInfo.calFirmwareName = "";
            entityInfo.processFileName = "";
            hzxmlSensor.GetChildText(sensorEntitys, "PartNumber", entityInfo.partNumber);
            hzxmlSensor.GetChildText(sensorEntitys, "SupplierCode", entityInfo.supplierCode);
            hzxmlSensor.GetChildText(sensorEntitys, "TargetVersion", entityInfo.targetVersion);

            XMLElement* file = hzxmlSensor.GetChildElement(sensorEntitys, "File");
            XMLElement* BootFirmware = hzxmlSensor.GetChildElement(file, "FlashDriverFirmware");
            if (hzxmlSensor.GetChildText(BootFirmware, "Name", entityInfo.flashDriverFirmwareName)){
                std::string flashDriverFirmwarePath = SENSOR_CONF_PATH + entityInfo.flashDriverFirmwareName;
                UPDATE_LOG_D("input flashDriverFirmwarePath: %s", flashDriverFirmwarePath.c_str());

                if (fileToBinHdl.Transition(flashDriverFirmwarePath, BIN_FILES_PATH)){
                    if ("" != fileToBinHdl.GetBinFileName(flashDriverFirmwarePath)){
                        entityInfo.flashDriverFirmwareName = fileToBinHdl.GetBinFileName(flashDriverFirmwarePath);
                    }
                    else{
                        entityInfo.flashDriverFirmwareName = flashDriverFirmwarePath;
                    }
                }
                else{
                    UPDATE_LOG_E("Transfer file : %s error!!!!!!!!!!!!", flashDriverFirmwarePath.c_str());
                    return -1;
                }
                UPDATE_LOG_D("flash flashDriverFirmwarePath: %s", entityInfo.flashDriverFirmwareName.c_str());
            }

            XMLElement* AppFirmware = hzxmlSensor.GetChildElement(file, "AppFirmware");
            if (hzxmlSensor.GetChildText(AppFirmware, "Name", entityInfo.appFirmwareName)){
                std::string appFirmwarePath = SENSOR_CONF_PATH + entityInfo.appFirmwareName;
                UPDATE_LOG_D("input appFirmwarePath: %s", appFirmwarePath.c_str());

                if (fileToBinHdl.Transition(appFirmwarePath, BIN_FILES_PATH)){
                    if ("" != fileToBinHdl.GetBinFileName(appFirmwarePath)){
                        entityInfo.appFirmwareName = fileToBinHdl.GetBinFileName(appFirmwarePath);
                    }
                    else{
                        entityInfo.appFirmwareName = appFirmwarePath;
                    }
                }
                else{
                    UPDATE_LOG_E("Transfer file : %s error!!!!!!!!!!!!", appFirmwarePath.c_str());
                    return -1;
                }
                UPDATE_LOG_D("flash appFirmwarePath: %s", entityInfo.appFirmwareName.c_str());
            }

            XMLElement* CalFile = hzxmlSensor.GetChildElement(file, "CalFirmware");
            if (hzxmlSensor.GetChildText(CalFile, "Name", entityInfo.calFirmwareName)){
                std::string calFirmwarePath = SENSOR_CONF_PATH + entityInfo.calFirmwareName;
                UPDATE_LOG_D("input calFirmwarePath: %s", calFirmwarePath.c_str());

                if (fileToBinHdl.Transition(calFirmwarePath, BIN_FILES_PATH)){
                    if ("" != fileToBinHdl.GetBinFileName(calFirmwarePath)){
                        entityInfo.calFirmwareName = fileToBinHdl.GetBinFileName(calFirmwarePath);
                    }
                    else{
                        entityInfo.calFirmwareName = calFirmwarePath;
                    }
                }
                else{
                    UPDATE_LOG_E("Transfer file : %s error!!!!!!!!!!!!", calFirmwarePath.c_str());
                    return -1;
                }
                UPDATE_LOG_D("flash calFirmwarePath: %s", entityInfo.calFirmwareName.c_str());
            }

            XMLElement* ProcessFile = hzxmlSensor.GetChildElement(file, "ProcessFile");
            if (hzxmlSensor.GetChildText(ProcessFile, "Name", entityInfo.processFileName)){
                std::string processFirmwarePath = SENSOR_CONF_PATH + entityInfo.processFileName;
                UPDATE_LOG_D("input processFirmwarePath: %s", processFirmwarePath.c_str());
                randomProcessFile = processFirmwarePath;
                entityInfo.processFileName = processFirmwarePath;
            }

            sensor.sensorInfo.entitys.push_back(entityInfo);
            sensorEntitys = hzxmlSensor.GetBrotherElement(sensorEntitys, "Entity");
            UM_DEBUG << "entityInfo.partNumber : " << entityInfo.partNumber 
                    << " , entityInfo.targetVersion : " << entityInfo.targetVersion 
                    << " , entityInfo.flashDriverFirmwareName : " << entityInfo.flashDriverFirmwareName 
                    << " , entityInfo.appFirmwareName : " << entityInfo.appFirmwareName 
                    << " , entityInfo.calFirmwareName : " << entityInfo.calFirmwareName 
                    << " , entityInfo.processFileName : " << entityInfo.processFileName; 
        }

        //parse process xml to update
        do {
            HzXml hzxmlProcess;
            // 这里的randomProcessFile只能随机选取一个entity的信息，因为初始还暂未确定使用哪一个
            bool ret = hzxmlProcess.ParseXmlFile(randomProcessFile.c_str());
            if (!ret) {
                UPDATE_LOG_D("praser process failed, ecu: %s, file: %s.", sensor.sensorInfo.name.c_str(), randomProcessFile.c_str());
                break;
            }

            XMLElement* rootProcess = hzxmlProcess.GetRootElement();
            hzxmlProcess.GetChildText(rootProcess, "Version", sensor.sensorInfo.processVersion);
            XMLElement* casex = hzxmlProcess.GetChildElement(rootProcess, "Case");
            if (nullptr == casex) {
                UPDATE_LOG_E("hzxmlProcess.GetChildElement get rootProcess case failed, file: %s", randomProcessFile.c_str());
                break;
            }
            while(casex)
            {
                UpdateCase_t    cases;
                cases.addrType = 0;
                cases.updateStep = "";
                cases.filePath = "";
                cases.fileType = 0;
                cases.memoryAddr = 0x00000000;
                cases.memorySize = 0x00000000;
                cases.recvDataSize = 0;
                cases.securityLevel = 0;
                cases.setcurityMask = 0x00000000;
                cases.transDataSize = 0;
                cases.transType = 0;
                cases.waitTime = 0;
                cases.delayTime = 0;
                cases.beginProgress = 0;
                cases.endProgress = 0;
                hzxmlProcess.GetChildText(casex, "TransType", strValue);
                cases.transType = "TransData"       == strValue  ? 1
                                : "SecurityAccess"  == strValue  ? 2
                                : "TransFile"       == strValue  ? 3
                                : 0;
                hzxmlProcess.GetChildText(casex, "UpdateStep", cases.updateStep);
                hzxmlProcess.GetChildText(casex, "AddrType", strValue);
                cases.addrType  = "PhysicAddr"      == strValue  ? 1
                                : "FunctionAddr"    == strValue  ? 2
                                : 0;
                hzxmlProcess.GetChildVector(casex, "TransDatas", cases.transData);
                hzxmlProcess.GetChildUintDec(casex, "TransLen", cases.transDataSize);
                hzxmlProcess.GetChildVector(casex, "RecvDatas", cases.recvExpect);
                hzxmlProcess.GetChildUintDec(casex, "RecvLen", cases.recvDataSize);
                hzxmlProcess.GetChildUintHex(casex, "Wait", cases.waitTime);
                hzxmlProcess.GetChildUintHex(casex, "Delay", cases.delayTime);
                hzxmlProcess.GetChildText(casex, "SecurityLevel", strValue);
                cases.securityLevel = "Level1"      == strValue  ? 1
                                    : "LevelFBL"    == strValue  ? 2
                                    : 0;
                hzxmlProcess.GetChildUintHex(casex, "SecurityMask", cases.setcurityMask);
                // <TransType>TransFile</TransType>
                // <FileType>CalFirmware</FileType>
                // <FileName>ussc_cal_vxxxx.hex</FileName>
                // <FileFormat>Hex</FileFormat>
                if (hzxmlProcess.GetChildText(casex, "FilePath", cases.filePath)){
                    hzxmlProcess.GetChildText(casex, "FileType", strValue);
                    cases.fileType      = "BootFirmware"   == strValue  ? 1
                                        : "AppFirmware"    == strValue  ? 2
                                        : "CalFirmware"    == strValue  ? 3
                                        : 0;
                    hzxmlProcess.GetChildUintHex(casex, "MemoryAddr", cases.memoryAddr);
                    hzxmlProcess.GetChildUintHex(casex, "MemorySize", cases.memorySize);


                    std::string tmpFilePath = SENSOR_CONF_PATH + cases.filePath;

                    if ("" != fileToBinHdl.GetBinFileName(tmpFilePath)){
                        UPDATE_LOG_D("in process xml, file: %s is s19 or hex file !", tmpFilePath.c_str());
                        cases.filePath = fileToBinHdl.GetBinFileName(tmpFilePath);
                        cases.memoryAddr = fileToBinHdl.GetFlashStartAddr(tmpFilePath);
                        cases.memorySize = fileToBinHdl.GetBinFileSize(tmpFilePath);

                    }
                    else{
                        UPDATE_LOG_D("in process xml, file: %s  is not a s19 or hex file !", tmpFilePath.c_str());
                        cases.filePath = tmpFilePath;
                        if (0x00000000 == cases.memorySize)
                        {
                            cases.memorySize = getFileSize(tmpFilePath);
                        }
                    }
                    UPDATE_LOG_D("in process xml, filePath:[%s], fileType:[%d], memoryAddr[%x], memorySize[%x]", cases.filePath.c_str(), cases.fileType, cases.memoryAddr, cases.memorySize);
                }

                if (hzxmlProcess.GetChildText(casex, "BeginProgress", strValue)) {
                    cases.beginProgress = strtof(strValue.c_str(), nullptr);
                }
                if (hzxmlProcess.GetChildText(casex, "EndProgress", strValue)) {
                    cases.endProgress = strtof(strValue.c_str(), nullptr);
                }

                // UPDATE_LOG_D ("process cases index: %ld, TransType: %d, AddrType: %d, TransDatas size: %ld, transLen: %d, RecvDatas size: %ld, recvLen: %d, wait: %d,securityLevel: %d, securityMask: %08X, fileName: %s, fileType: %d, fileFormat: %d, memoryAddr: %08X, memorySize: %08X",
                //     updateProcess.process.size(), cases.transType, cases.addrType, cases.transData.size(), cases.transDataSize, cases.recvExpect.size(), cases.recvDataSize, cases.waitTime, cases.securityLevel, cases.setcurityMask,
                //     cases.fileName.c_str(), cases.fileType, cases.fileFormat, cases.memoryAddr, cases.memorySize);

                sensor.process.push_back(cases);

                casex = hzxmlProcess.GetBrotherElement(casex, "Case");
            }

        } while (0);

        sensor_manifest_.sensors.push_back(sensor);
        sensorx = hzxmlSensor.GetBrotherElement(sensorx, "Sensor");

        UPDATE_LOG_D("add sensor[%ld] name: %s, updateType: %d, canidTx: %X, canidRx: %X, logicalAddr: %X, functionAddr: %X, updateSequence: %d, processSize: %ld.",
            sensor_manifest_.sensors.size(), sensor.sensorInfo.name.c_str(), sensor.sensorInfo.updateType, sensor.sensorInfo.canidTx, sensor.sensorInfo.canidRx, sensor.sensorInfo.logicalAddr, sensor.sensorInfo.functionAddr,
            sensor.sensorInfo.updateSequence, sensor.process.size());
    }

    UPDATE_LOG_D("ParseSensorManifest successful, sensor size: %ld.", sensor_manifest_.sensors.size());
    return 0;
}

int32_t
ConfigManager::ParseSocManifest()
{
    UPDATE_LOG_D("ParseSocManifest");
    //解析socsor_manifest.xml放到soc_manifest_
    soc_manifest_.processProportion = "";
    soc_manifest_.version = "";
    soc_manifest_.socs.clear();
    version_info_.socVersion = "";
    version_info_.mcuVersion = "";
    version_info_.dsvVersion = "";

    std::string socManifestFile = SOC_CONF_FILE_PATH + SOC_MANIFEST_NAME;
    UPDATE_LOG_D("socManifestFile: %s", socManifestFile.c_str());

    if (!PathExists(socManifestFile)){
        UPDATE_LOG_W("soc manifest xml not Exists");
        return -2;
    }
    HzXml hzxmlSoc;
    std::string strValue;
    if (!hzxmlSoc.ParseXmlFile(socManifestFile.c_str())) {
        UPDATE_LOG_E("ParseXmlFile soc_manifest.xml failed");
        return -1;
    }

    XMLElement* root = hzxmlSoc.GetRootElement();
    hzxmlSoc.GetChildText(root, "Version", soc_manifest_.version);
    hzxmlSoc.GetChildText(root, "ProcessProportion", soc_manifest_.processProportion);
    XMLElement* socx = hzxmlSoc.GetChildElement(root, "Soc");
    while(socx)
    {
        SoC_t     soc;
        soc.socInfo.name = "";
        soc.socInfo.updateType = 0;
        soc.socInfo.havaFileSystem = false;
        soc.socInfo.partNumber = "";
        soc.socInfo.supplierCode = "";
        soc.socInfo.targetVersion = "";

        hzxmlSoc.GetChildText(socx, "Name", soc.socInfo.name);
        hzxmlSoc.GetChildText(socx, "UpdateType", strValue);
        soc.socInfo.updateType = "docan" == strValue ? 1
                              : "doip"  == strValue ? 2
                              : "interface" == strValue ? 3
                              : "someip" == strValue ? 4
                              : 0;
        if (hzxmlSoc.GetChildText(socx, "LogicalAddr", strValue)) {
            soc.socInfo.logicalAddr = std::stoi(strValue, 0, 16);
        }
        hzxmlSoc.GetChildText(socx, "HavaFileSystem", strValue);
        soc.socInfo.havaFileSystem = "True" == strValue ? true
                                   : "true" == strValue ? true
                                   : false;
        hzxmlSoc.GetChildText(socx, "SameVersionCheck", strValue);
        soc.socInfo.sameVersionCheck = "False" == strValue ? false
                                     : "false" == strValue ? false
                                     : true;
        hzxmlSoc.GetChildText(socx, "PartNumber", soc.socInfo.partNumber);
        hzxmlSoc.GetChildText(socx, "SupplierCode", soc.socInfo.supplierCode);
        hzxmlSoc.GetChildText(socx, "TargetAppVersion", soc.socInfo.targetVersion);
        hzxmlSoc.GetChildText(socx, "TargetAppVersion", version_info_.socVersion);
        hzxmlSoc.GetChildText(socx, "TargetMcuVersion", version_info_.mcuVersion);
        hzxmlSoc.GetChildText(socx, "TargetDsvVersion", version_info_.dsvVersion);

        XMLElement* file = hzxmlSoc.GetChildElement(socx, "File");
        XMLElement* firmware = hzxmlSoc.GetChildElement(file, "Firmware");
        if (hzxmlSoc.GetChildText(firmware, "Name", soc.socInfo.firmwareName)){
            if (UpdateSettings::Instance().UseDoubleCompress()) {
                soc.socInfo.firmwareName = PAKAGE_UNZIP_PATH + soc.socInfo.firmwareName;
            } else {
                soc.socInfo.firmwareName = PAKAGE_UNZIP_PATH;
            }
        }
        UM_DEBUG << "deasy update param is : " << soc.socInfo.firmwareName;
        soc_manifest_.socs.push_back(soc);
        socx = hzxmlSoc.GetBrotherElement(socx, "Soc");

        UPDATE_LOG_D("add soc[%ld] name: %s, updateType: %d, partNumber: %s, supplierCode: %s, targetVersion: %s, Firmware: %s.",
            soc_manifest_.socs.size(), soc.socInfo.name.c_str(), soc.socInfo.updateType, soc.socInfo.partNumber.c_str(),
            soc.socInfo.supplierCode.c_str(), soc.socInfo.targetVersion.c_str(), soc.socInfo.firmwareName.c_str());
        UM_DEBUG << "TargetSocVersion is : " << version_info_.socVersion;
        UM_DEBUG << "TargetMcuVersion is : " << version_info_.mcuVersion;
        UM_DEBUG << "TargetDsvVersion is : " << version_info_.dsvVersion;

    }

    UPDATE_LOG_D("ParseSocManifest successful, soc size: %ld.", soc_manifest_.socs.size());
    return 0;
}

bool
ConfigManager::GetVin(std::vector<uint8_t>& vin)
{
    // TBD. get 17 bytes vin
    vin = { 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37 };
    return true;
}

bool
ConfigManager::GetDate(std::vector<uint8_t>& date)
{
    // TBD. get 4 bytes dates for 2022/01/09  --> { 0x20, 0x22, 0x01, 0x09 }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm* timeinfo = localtime(&ts.tv_sec);
    char time_buf[128] = { 0 };
    snprintf(time_buf, sizeof(time_buf) - 1, "%04d%02d%02d",
        timeinfo->tm_year + 1900,
        timeinfo->tm_mon + 1,
        timeinfo->tm_mday);
    for (uint8_t index = 0; index < strlen(time_buf); index += 2) {
        date.push_back((uint8_t)(((uint8_t)(time_buf[index] - '0') << 4) | (uint8_t)(time_buf[index + 1] - '0')));
    }
    return true;
}

bool
ConfigManager::GetTesterSN(std::vector<uint8_t>& sn)
{
    // TBD. get 10 bytes tester SN
    sn = { 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x30 };
    return true;
}

std::unordered_map<std::string, std::string>&
ConfigManager::GetFileList()
{
    return update_filelist_map_;
}

UpdateConfig_t&
ConfigManager::GetUpdateConfig()
{
    return update_config_;
}

SensorManifest_t&
ConfigManager::GetSensorManifest()
{
    return sensor_manifest_;
}

bool
ConfigManager::GetSensorManifestByName(const std::string& sensorName, Sensor_t& info)
{
    for (auto it : sensor_manifest_.sensors) {
        if (it.sensorInfo.name == sensorName) {
            UM_DEBUG << "GetSensorManifestByName succ, sensorName is : " << sensorName;
            info = it;
            break;
        }
    }
    return true;
}


SocManifest_t&
ConfigManager::GetSocManifest()
{
    return soc_manifest_;
}

bool 
ConfigManager::GetMajorVersion(std::string& majorVersion)
{
    UPDATE_LOG_D("ConfigManager::GetMajorVersion.");
    if (version_info_.majorVersion == "")
    {
        UPDATE_LOG_E("GetMajorVersion error, it is empty.");
        return false;
    }
    UPDATE_LOG_D("major version is : %s", version_info_.majorVersion.c_str());
    majorVersion = version_info_.majorVersion;
    return true;
}

bool 
ConfigManager::GetSocVersion(std::string& socVersion)
{
    UPDATE_LOG_D("ConfigManager::GetSocVersion.");
    if (version_info_.socVersion == "")
    {
        UPDATE_LOG_E("GetSocVersion error, it is empty.");
        return false;
    }
    UPDATE_LOG_D("soc version is : %s", version_info_.socVersion.c_str());
    socVersion = version_info_.socVersion;
    return true;
}

bool 
ConfigManager::GetMcuVersion(std::string& mcuVersion)
{
    UPDATE_LOG_D("ConfigManager::GetMcuVersion.");
    if (version_info_.mcuVersion == "")
    {
        UPDATE_LOG_E("GetMcuVersion error, it is empty.");
        return false;
    }
    UPDATE_LOG_D("mcu version is : %s", version_info_.mcuVersion.c_str());
    mcuVersion = version_info_.mcuVersion;
    return true;
}

bool 
ConfigManager::GetDsvVersion(std::string& dsvVersion)
{
    UPDATE_LOG_D("ConfigManager::GetDsvVersion.");
    if (version_info_.dsvVersion == "")
    {
        UPDATE_LOG_E("GetDsvVersion error, it is empty.");
        return false;
    }
    UPDATE_LOG_D("dsv version is : %s", version_info_.dsvVersion.c_str());
    dsvVersion = version_info_.dsvVersion;
    return true;
}

bool 
ConfigManager::MountSensors()
{
    UPDATE_LOG_D("ConfigManager::MountSensors.");
    
    bool bRet{false};
    bRet = PathCreate(PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    if(!bRet)
    {
        UPDATE_LOG_E("create path error");
        return false;
    }
    if (UpdateSettings::Instance().UseDoubleCompress()) {
        bRet = Ecb_decryptFile(PAKAGE_UNZIP_PATH + AES_SENSOR_IMG_PATH, PAKAGE_UNZIP_PATH + SENSOR_IMG_PATH, aes_key);
    } else {
        bRet = Ecb_decryptFile(PAKAGE_UNZIP_PATH + AES_SENSOR_IMG_PATH_NEW, PAKAGE_UNZIP_PATH + SENSOR_IMG_PATH_NEW, aes_key);
    }
    if(!bRet)
    {
        UPDATE_LOG_E("aes decrypt error");
        return false;
    }
    if (UpdateSettings::Instance().UseDoubleCompress()) {
        bRet = FileMount(PAKAGE_UNZIP_PATH + SENSOR_IMG_PATH, PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    } else {
        bRet = FileMount(PAKAGE_UNZIP_PATH + SENSOR_IMG_PATH_NEW, PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    }
    if(!bRet)
    {
        UPDATE_LOG_W("mount sensor error");
        return false;
    }
    return true;
}

bool 
ConfigManager::UmountSensors()
{
    UPDATE_LOG_D("ConfigManager::UmountSensors.");
    
    bool bRet{false};

    bRet = FileUmount(PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    if(!bRet)
    {
        UPDATE_LOG_W("mount sensor error");
        return false;
    }
    return true;
}

bool 
ConfigManager::UmountAndRemoveSensors()
{
    UPDATE_LOG_D("ConfigManager::UmountAndRemoveSensors.");
    
    bool bRet{false};

    bRet = FileUmount(PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    if(!bRet)
    {
        UPDATE_LOG_W("mount sensor error");
    }
    bRet = PathRemove(PAKAGE_UNZIP_PATH + SENSOR_MOUNT_TARGET);
    if(!bRet)
    {
        UPDATE_LOG_W("remove sensor error");
        return false;
    }
    return true;
}

bool 
ConfigManager::Ecb_decryptFile(const std::string& inputFilePath, const std::string& outputFilePath, const std::string& key) {
    UM_DEBUG << "ConfigManager::Ecb_decryptFile.";
    std::ifstream inputFile(inputFilePath, std::ios::binary);
    std::ofstream outputFile(outputFilePath, std::ios::out);

    if (!inputFile.is_open() || !outputFile.is_open()) {
        UM_ERROR << "Error opening files.";
        return false;
    }

    unsigned char inbuffer[1024];
    unsigned char outbuffer[2048];

    int outlen = 0;

    OpenSSL_add_all_algorithms();  // 可选，添加所有算法

    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        UM_ERROR << "EVP_CIPHER_CTX_new failed.";
        return false;
    }

    // 设置AES解密密钥
    const EVP_CIPHER *cipher = EVP_aes_256_ecb();
    unsigned char pkey[32];

    if (key.length() < sizeof(pkey)) {
        UM_ERROR << "Key size error.";
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    key.copy(reinterpret_cast<char*>(pkey), sizeof(pkey), 0);

    if (EVP_DecryptInit_ex(ctx, cipher, nullptr, pkey, nullptr) != 1) {
        UM_ERROR << "EVP_DecryptInit_ex failed.";
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    while (true) {
        inputFile.read(reinterpret_cast<char*>(inbuffer), 1024);
        int bytesRead = inputFile.gcount();
        if (bytesRead <= 0) {
            break;
        }

        if (EVP_DecryptUpdate(ctx, outbuffer, &outlen, inbuffer, bytesRead) != 1) {
            UM_ERROR << "EVP_DecryptUpdate failed.";
            EVP_CIPHER_CTX_free(ctx);
            return false;
        }

        outputFile.write(reinterpret_cast<const char*>(outbuffer), outlen);
    }

    int final_len = 0;
    if (EVP_DecryptFinal_ex(ctx, outbuffer, &final_len) != 1) {
        UM_ERROR << "EVP_DecryptFinal_ex failed.";
        EVP_CIPHER_CTX_free(ctx);
        return false;
    }

    UM_DEBUG << "final_len:" << final_len;
    outputFile.write(reinterpret_cast<const char*>(outbuffer), final_len);

    EVP_CIPHER_CTX_free(ctx);
    inputFile.close();
    outputFile.close();

    UM_DEBUG << "File decrypted successfully.";
    return true;
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon