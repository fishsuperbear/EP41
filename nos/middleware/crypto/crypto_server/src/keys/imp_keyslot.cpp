#include "keys/imp_keyslot.h"

#include <memory>
#include <string>
#include <vector>
#include <ios>
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <openssl/pem.h>
#include <filesystem>

#include "common/imp_io_interface.h"
#include "common/crypto_logger.hpp"
#include "cryp/imp_crypto_provider.h"
#include "server/crypto_server_config.h"
#include "server/encryption_service.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

netaos::core::Result<void> ImpKeySlot::Clear() noexcept{
    contentProps_.mAlgId = crypto::kAlgIdUndefined;
    contentProps_.mObjectSize = 0;
    contentProps_.mObjectType = CryptoObjectType::kUndefined;
    contentProps_.mObjectUid.mGeneratorUid.mQwordLs = 0;
    contentProps_.mObjectUid.mGeneratorUid.mQwordMs = 0;
    contentProps_.mObjectUid.mVersionStamp = 0;
    return netaos::core::Result<void>();
};

netaos::core::Result<KeySlotContentProps> ImpKeySlot::GetContentProps() const noexcept {
    return netaos::core::Result<KeySlotContentProps>::FromValue(contentProps_);
};


netaos::core::Result<cryp::CryptoProvider::Uptr> ImpKeySlot::MyProvider() const noexcept{
    auto uptr = std::make_unique<cryp::ImpCryptoProvider>();
    return netaos::core::Result<cryp::CryptoProvider::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<KeySlotPrototypeProps> ImpKeySlot::GetPrototypedProps() const noexcept {
    return netaos::core::Result<KeySlotPrototypeProps>::FromValue(protoProps_);
};

bool ImpKeySlot::IsEmpty() const noexcept {
    CRYP_INFO<<"contentProps_.mAlgId:" <<contentProps_.mAlgId;
    CRYP_INFO<<"contentProps_.mObjectUid.IsNil():"<<contentProps_.mObjectUid.IsNil();
    if((contentProps_.mAlgId == crypto::kAlgIdUndefined) && (contentProps_.mObjectUid.IsNil())) {
        return true;
    }else {
        return false;
    }
};

netaos::core::Result<IOInterface::Uptr> ImpKeySlot::Open(bool subscribeForUpdates, bool writeable) const noexcept {
    std::string secPath = CryptoConfig::Instance().GetKeysStoragePath();
    std::string slotnum = std::to_string(protoProps_.mslotNum);
    std::string slotfile = secPath + slotnum + "-" + protoProps_.mslotUuid.ToUuidStr() + ".json.encrypted";

    std::vector<std::uint8_t> pay;
    std::ifstream ifs(slotfile);
    bool existFile = std::filesystem::exists(slotfile);

    CRYP_INFO<<"ImpKeySlot open, filePath:"<<slotfile << "; exist file ? :" << static_cast<int>(existFile);

    if(existFile){
        Json::Value jsonData;
        Json::Reader reader;
        std::string fileContent = "";
        if(!EncryptionService::Instance().do_FileCrypt_ReturnString(slotfile, fileContent, 0)) {
            CRYP_ERROR<<"Open do_FileCrypt_ReturnString error ";
            return netaos::core::Result<IOInterface::Uptr>::FromValue(std::move(std::make_unique<ImpIOInterface>(crypto::ImpIOInterface::IOInterfaceInfo(),pay)));
        }
        // std::string fileContent((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        bool parsingSuccessful = reader.parse(fileContent, jsonData);
        if (!parsingSuccessful) {
            CRYP_ERROR<<"ImpKeySlot open, json parse error !";
            ifs.close();
            return netaos::core::Result<IOInterface::Uptr>::FromValue(std::move(std::make_unique<ImpIOInterface>(crypto::ImpIOInterface::IOInterfaceInfo(),pay)));
        }
        // ifs >> jsonData;
        std::string keyData = "";
        // 检索 "KeyData" 的值
        if (jsonData.isObject() &&
            jsonData.isMember("value") &&
            jsonData["value"].isObject() &&
            jsonData["value"].isMember("object[]") &&
            jsonData["value"]["object[]"].isArray() &&
            jsonData["value"]["object[]"].size() > 5 &&
            jsonData["value"]["object[]"][5].isObject() &&
            jsonData["value"]["object[]"][5].isMember("value") &&
            jsonData["value"]["object[]"][5]["value"].isObject() &&
            jsonData["value"]["object[]"][5]["value"].isMember("string")) {
            keyData = jsonData["value"]["object[]"][5]["value"]["string"].asString();
            // 使用 keyData 值进行后续操作
        } else {
            CRYP_ERROR<<"ImpKeySlot open, jsonData is null !";
            ifs.close();
            return netaos::core::Result<IOInterface::Uptr>::FromValue(std::move(std::make_unique<ImpIOInterface>(crypto::ImpIOInterface::IOInterfaceInfo(),pay)));
        }
        // std::string keyData = jsonData["value"]["object[]"][5]["value"]["string"].asString();
        // 输出 "KeyData" 的值
        // std::cout << "ImpKeySlot Open KeyData: " << std::endl;
        pay.assign(keyData.begin(), keyData.end());
        ifs.close();
    }
    crypto::ImpIOInterface::IOInterfaceInfo  ioInfo;
    ioInfo.isWritable = writeable;
    ioInfo.isExportable = protoProps_.mExportAllowed;
    // ioInfo.isSession = false;  // TODO how to set it
    // ioInfo.objectSize = contentProps_.mObjectSize;
    ioInfo.capacity = protoProps_.mSlotCapacity;
    ioInfo.usage = protoProps_.mContentAllowedUsage;
    ioInfo.algId = protoProps_.mAlgId;
    // ioInfo.objectUid = contentProps_.mObjectUid;
    // ioInfo.objectType = contentProps_.mObjectType;

    auto uptr = std::make_unique<ImpIOInterface>(ioInfo,pay);
    return netaos::core::Result<IOInterface::Uptr>::FromValue(std::move(uptr));
};

netaos::core::Result<void> ImpKeySlot::SaveCopy(const IOInterface& container) noexcept {
    // check key property
    CRYP_INFO << "save copy slot uuid: " << protoProps_.mslotUuid.ToUuidStr();
    if (protoProps_.mObjectType != dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.objectType) {
        CRYP_ERROR << "key slot object type:" << static_cast<int>(protoProps_.mObjectType)
                   << " key object type:" << static_cast<int>(dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.objectType);
        return netaos::core::Result<void>();
    }
    if (protoProps_.mContentAllowedUsage != dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.usage) {
        CRYP_ERROR << "key slot allowed usage:" << static_cast<int>(protoProps_.mContentAllowedUsage)
                   << " key allowed usage:" << static_cast<int>(dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.usage);
        return netaos::core::Result<void>();
    }
    if (protoProps_.mSlotCapacity < dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.objectSize) {
        CRYP_ERROR << "key slot capacity:" << static_cast<int>(protoProps_.mSlotCapacity)
                   << " key capacity:" << static_cast<int>(dynamic_cast<ImpIOInterface&>(const_cast<IOInterface&>(container)).ioInfo_.objectSize);
        return netaos::core::Result<void>();
    }

    const ImpIOInterface* impl = dynamic_cast<const ImpIOInterface*>(&container);
    content_ = std::make_shared<ImpIOInterface>(*impl);
    return netaos::core::Result<void>();
};

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara