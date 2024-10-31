#include "keys/imp_key_storage_provider.h"

#include "common/crypto_error_domain.h"
#include "common/imp_io_interface.h"
#include "common/crypto_logger.hpp"

#include "keys/elementary_types.h"
#include "keys/imp_keyslot.h"
#include "keys/json_parser.h"

#include "server/encryption_service.hpp"
#include "server/crypto_server_config.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

std::vector<std::tuple<TransactionId, std::uint32_t, std::string ,std::string>> transaction;

std::string createSavekeyFile(std::shared_ptr<IOInterface> iOInterface, KeySlotPrototypeProps protoTypeProps ,std::string keyData) {
    if (!iOInterface) {
        return "";
    }
    Json::Value data;
     // 添加数据到 JSON 对象
    data["key"] = "KeySlotContentProps";
    data["value"]["object[]"][0]["key"] = "ObjectUid";
    data["value"]["object[]"][0]["value"]["string"] = iOInterface->GetObjectId().mGeneratorUid.ToUuidStr();
    data["value"]["object[]"][1]["key"] = "AlgId";
    data["value"]["object[]"][1]["value"]["uint64"] = protoTypeProps.mAlgId;
    data["value"]["object[]"][2]["key"] = "ObjectSize";
    data["value"]["object[]"][2]["value"]["uint64"] = iOInterface->GetPayloadSize();
    data["value"]["object[]"][3]["key"] = "AllowedUsage";
    data["value"]["object[]"][3]["value"]["uint32"] = iOInterface->GetAllowedUsage();
    data["value"]["object[]"][4]["key"] = "ObjectType";
    data["value"]["object[]"][4]["value"]["uint32"] = static_cast<int>(protoTypeProps.mObjectType);
    data["value"]["object[]"][5]["key"] = "KeyData";
    data["value"]["object[]"][5]["value"]["string"] = keyData;


    // 构建 JSON 写入器
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "  ";

    // 构造文件名
    std::string keysPath = CryptoConfig::Instance().GetKeysStoragePath();
    std::string slotnum = std::to_string(protoTypeProps.mslotNum);
    std::string slotUuid = protoTypeProps.mslotUuid.ToUuidStr();
    std::string keyFile = keysPath + slotnum + "-" + slotUuid + ".json" + ".tmp";

    // 打开文件
    std::ofstream ofs(keyFile);

    // 在输出流上设置宽度
    ofs << std::setw(2);

    // 将输出流与文件关联
    writer.newStreamWriter()->write(data, &ofs);

    // 关闭文件
    ofs.close();

    return keyFile;
}

netaos::core::Result<TransactionId> IMPKeyStorageProvider::BeginTransaction(const TransactionScope& targetSlots) noexcept {
    static TransactionId id;
    id++;
    std::string slotnum;
    std::string slotUuid;
    std::string keysPath = CryptoConfig::Instance().GetKeysStoragePath();
    // std::vector<uint8_t> pay;
    for (auto slot : targetSlots) {
        slotnum = std::to_string(slot->GetPrototypedProps().Value().mslotNum);
        if (!slot) {
            CRYP_ERROR<<"error BeginTransaction slot is null !";
            return netaos::core::Result<TransactionId>::FromValue(id);
        }
        auto impslot = dynamic_cast<ImpKeySlot*>(slot);
        if (!impslot) {
            CRYP_ERROR<<"error BeginTransaction impslot is null !";
            return netaos::core::Result<TransactionId>::FromValue(id);
        }
        std::string key_Data(reinterpret_cast<const char*>(impslot->GetContent()->GetPayload().data()), static_cast<std::streamsize>(impslot->GetContent()->GetPayload().size()));
        std::string keyFile = createSavekeyFile(impslot->GetContent(), slot->GetPrototypedProps().Value(), key_Data);
        transaction.push_back(std::make_tuple(id, slot->GetPrototypedProps().Value().mslotNum, slot->GetPrototypedProps().Value().mslotUuid.ToUuidStr(), keyFile));
        CRYP_INFO<<"BeginTransaction successed !, temp file: "<<keyFile;
    }
    return netaos::core::Result<TransactionId>::FromValue(id);
};

netaos::core::Result<void> IMPKeyStorageProvider::CommitTransaction(TransactionId id) noexcept{
    for (auto ts : transaction) {
        std::string decrypted_key_file;
        std::string encrypted_key_file;
        if (id == std::get<0>(ts)) {
            decrypted_key_file = CryptoConfig::Instance().GetKeysStoragePath() + std::to_string(std::get<1>(ts)) + "-" + std::get<2>(ts) + ".json";
            encrypted_key_file = CryptoConfig::Instance().GetKeysStoragePath() + std::to_string(std::get<1>(ts)) + "-" + std::get<2>(ts) + ".json.encrypted";
            if (0 == std::rename(std::get<3>(ts).c_str(), decrypted_key_file.c_str())) {
                CRYP_INFO<<"CommitTransaction successed!, raw file :"<< decrypted_key_file;
            }
            // 加密key文件
            if(!EncryptionService::Instance().do_FileCrypt(decrypted_key_file, encrypted_key_file, 1)){
                CRYP_ERROR << "CommitTransaction do_FileCrypt 1 error";
                return netaos::core::Result<void>();
            }
            CRYP_INFO<<"CommitTransaction successed!, encrypted file :"<< encrypted_key_file;
            // 删除源文件
            if (std::remove(decrypted_key_file.c_str()) != 0) {
                CRYP_ERROR << "Unable to delete file 'example.txt'";
            }
            // //解密文件 test ok
            // std::string decrypted_key_file_test = decrypted_key_file;
            // if(!EncryptionService::Instance().do_FileCrypt(encrypted_key_file, decrypted_key_file_test, 0)) {
            //     CRYP_ERROR << "CommitTransaction do_FileCrypt 0 error";
            //     return netaos::core::Result<void>();
            // }
            // CRYP_INFO<<"CommitTransaction successed!, decrypted file :"<< decrypted_key_file_test;
        }
    }
    return netaos::core::Result<void>();
};

// UpdatesObserver::Uptr IMPKeyStorageProvider::GetRegisteredObserver() const noexcept{
// };

netaos::core::Result<KeySlot::Uptr> IMPKeyStorageProvider::LoadKeySlot(netaos::core::InstanceSpecifier& iSpecify) noexcept{
    KeySlotPrototypeProps protoProps;
    KeySlotContentProps contentProps;
    netaos::core::StringView spec = iSpecify.ToString();
    size_t underline_pos = spec.find(".json");
    netaos::core::StringView json_sv = spec.substr(0, underline_pos + 5);
    std::string jsonFile (json_sv.data(),json_sv.size());
    std::string uuid = spec.substr(underline_pos + 6, spec.size() - jsonFile.size() - 1).data();

    CRYP_INFO << "jsonFile:" << jsonFile ;
    CRYP_INFO << "uuid:" << uuid;

    if (!JsonParser::parseJson(jsonFile,uuid, protoProps, contentProps)) {
        return hozon::netaos::core::Result<KeySlot::Uptr>::FromError(hozon::netaos::crypto::MakeErrorCode(hozon::netaos::crypto::CryptoErrc::kUnknownIdentifier, 0));
    }
    std::unique_ptr<ImpKeySlot> uptr(new ImpKeySlot(protoProps, contentProps));
    return netaos::core::Result<KeySlot::Uptr>::FromValue(std::move(uptr));
};

// UpdatesObserver::Uptr IMPKeyStorageProvider::RegisterObserver(UpdatesObserver::Uptr observer = nullptr) noexcept{
// };

netaos::core::Result<void> IMPKeyStorageProvider::RollbackTransaction(TransactionId id) noexcept{

    return netaos::core::Result<void>();
};

netaos::core::Result<void> IMPKeyStorageProvider::UnsubscribeObserver(KeySlot& slot) noexcept{

    return netaos::core::Result<void>();
};

}  // namespace KEYS
}  // namespace crypto
}  // namespace ara
}  // namespace ara