
#include "keys/json_parser.h"
#include "common/crypto_logger.hpp"

namespace hozon {
namespace netaos {
namespace crypto {
namespace keys {

crypto::Uuid JsonParser::convertStringtoUuid(std::string uuidStr){
    crypto::Uuid uuid(0ull,0ull);
    std::string ls,ms;
    // std::size_t lsLen,msLen;
    if (!uuidStr.empty()) {
        uuidStr.erase(std::remove(uuidStr.begin(), uuidStr.end(), '-'), uuidStr.end());
        // 去除高16位
        ms = uuidStr.substr(0, 16);
        // 去除低16位
        ls = uuidStr.substr(16, 16);
        uuid.mQwordLs = strtoull(ls.data(),0,16);
        uuid.mQwordMs = strtoull(ms.data(),0,16);
        CRYP_INFO<<"ls:"<<ls<<" ms:"<<ms;
        CRYP_INFO<<"uuid.mQwordLs:"<<uuid.mQwordLs<<"  uuid..mQwordMs:"<<uuid.mQwordMs;
    }

    return uuid;
}

crypto::CryptoObjectType JsonParser::convertStringtoCryptoObjectType(std::string str){
    crypto::CryptoObjectType objType = CryptoObjectType::kUndefined;
    if(!str.compare("PRIVATE-KEY")){
        objType = CryptoObjectType::kPrivateKey;
    }else if(!str.compare("PUBLIC-KEY")){
        objType = CryptoObjectType::kPublicKey;
    }else if(!str.compare("SYMMETRIC-KEY")){
        objType = CryptoObjectType::kSymmetricKey;
    }else if(!str.compare("SIGNATURE")){
        objType = CryptoObjectType::kSignature;
    }else if(!str.compare("SECRETSEED")){
        objType = CryptoObjectType::kSecretSeed;
    }else if(!str.compare("DIAG-27")){
        objType = CryptoObjectType::kSecretSeed;
    }else{
        objType = CryptoObjectType::kUndefined;
    }
    return objType;
}

crypto::CryptoAlgId JsonParser::convertStringtoAlgId(std::string str){
    crypto::CryptoAlgId algid = kAlgIdUndefined;
    if(!str.compare("RSA2048-SHA384-PSS")){
        algid = kAlgIdRSA2048SHA384PSS;
    }else if(!str.compare("RSA2048-SHA512-PSS")){
        algid = kAlgIdRSA2048SHA512PSS;
    }else if(!str.compare("RSA3072-SHA256-PSS")){
        algid = kAlgIdRSA3072SHA256PSS;
    }else if(!str.compare("RSA3072-SHA512-PSS")){
        algid = kAlgIdRSA3072SHA512PSS;
    }else if(!str.compare("RSA4096-SHA256-PSS")){
        algid = kAlgIdRSA4096SHA256PSS;
    }else if(!str.compare("RSA4096-SHA384-PSS")){
        algid = kAlgIdRSA4096SHA384PSS;
    }else if(!str.compare("RSA2048-SHA384-PKCS")){
        algid = kAlgIdRSA2048SHA384PKCS;
    }else if(!str.compare("RSA2048-SHA512-PKCS")){
        algid = kAlgIdRSA2048SHA512PKCS;
    }else if(!str.compare("RSA3072-SHA256-PKCS")){
        algid = kAlgIdRSA3072SHA256PKCS;
    }else if(!str.compare("RSA3072-SHA256-PKCS")){
        algid = kAlgIdRSA3072SHA512PKCS;
    }else if(!str.compare("RSA4096-SHA256-PKCS")){
        algid = kAlgIdRSA4096SHA256PKCS;
    }else if(!str.compare("RSA4096-SHA384-PKCS")){
        algid = kAlgIdRSA4096SHA384PKCS;
    }else if(!str.compare("RSA4096-SHA384-PKCS")){
        algid = kAlgIdRSA4096SHA384PKCS;
    }else if(!str.compare("CBC-AES-256")){
        algid = kAlgIdCBCAES256;
    }else if (!str.compare("RSA2048-SHA256-PKCS")) {
        algid = kAlgIdRSA2048SHA256PSS;
    }else{

    }
    return algid;
}

bool JsonParser::parseJson(std::string jsonfile,std::string& uuid,KeySlotPrototypeProps& protoProps,KeySlotContentProps& contentProps){

    bool ret = false;
    Json::CharReaderBuilder readBuilder;
    std::unique_ptr<Json::CharReader> reader(readBuilder.newCharReader());
    Json::Value root;
    Json::String errs;
    CRYP_INFO<<"parseJson jsonfile:"<<jsonfile << "  uuid : "<< uuid;
    std::ifstream ifs(jsonfile.data());
    Json::Value protoPro,contentPro;
    // bool matchuuid = false;

    if (!ifs.is_open()) {
        CRYP_ERROR<< "ifs is null";
        return false;
    }

    try {
        if (!Json::parseFromStream(readBuilder, ifs, &root, &errs)) {
            CRYP_ERROR<<"parseFromStream failed, errs: "<<errs.c_str();
            ifs.close();
            return false;
        }{
            CRYP_INFO<< "parseFromStream success.";
        }

        CRYP_INFO<<"root.type:"<<root.type() << " root.size" << root.size();
        if(root.type() == Json::arrayValue){
            for (unsigned int i = 0; i < root.size(); i++) {
                CRYP_INFO << root[i]["key"].asString() << " "<< uuid.data();
                if (!root[i]["key"].asString().compare(uuid.data())) {  //find uuid
                    for(unsigned int j = 0; j < root[i]["value"]["object[]"].size(); j++){
                        if(!root[i]["value"]["object[]"][j]["key"].asString().compare("KeySlotPrototypeProps")){// find protoProps
                            CRYP_INFO<< "get protoPro";
                            protoPro = root[i]["value"]["object[]"][j]["value"]["object[]"];
                            break;
                        }
                    }

                    for(unsigned int k = 0; k < root[i]["value"]["object[]"].size(); k++){
                        if(!root[i]["value"]["object[]"][k]["key"].asString().compare("KeySlotContentProps")){  // find protoProps
                            CRYP_INFO<< "get contentPro";
                            contentPro = root[i]["value"]["object[]"][k]["value"]["object[]"];
                            break;
                        }
                    }
                    break;
                }
            }
        }else{
            ifs.close();
            CRYP_ERROR<< "json body format error.";
            return ret;
        }

        CRYP_INFO<<"protoPro.type:"<<protoPro.type();
        if(protoPro && (protoPro.type() == Json::arrayValue)) {
            CRYP_INFO<<"protoPro.size:"<<protoPro.size();
            for(unsigned int i=0;i<protoPro.size();i++){
                if(!protoPro[i]["key"].asString().compare("SlotUid")){
                    protoProps.mslotUuid = FromString(protoPro[i]["value"]["string"].asString()).Value();
                    CRYP_INFO << "mslotUuid: " << protoProps.mslotUuid.ToUuidStr();
                }else if(!protoPro[i]["key"].asString().compare("SlotNumber")){
                    protoProps.mslotNum = protoPro[i]["value"]["uint32"].asUInt();
                    CRYP_INFO<<"mslotNum:"<< protoProps.mslotNum ;
                }else if(!protoPro[i]["key"].asString().compare("KeySlotType")){
                    protoProps.mSlotType = static_cast<KeySlotType>(protoPro[i]["value"]["uint32"].asUInt());
                    CRYP_INFO<<"KeySlotType:"<<static_cast<std::uint32_t>(protoProps.mSlotType);
                }else if(!protoPro[i]["key"].asString().compare("CryptoAlgType")){
                    protoProps.mAlgId =  convertStringtoAlgId(protoPro[i]["value"]["string"].asString());
                    CRYP_INFO<<"CryptoAlgType:"<<protoProps.mAlgId;
                }else if(!protoPro[i]["key"].asString().compare("CryptoObjectType")){
                    protoProps.mObjectType = convertStringtoCryptoObjectType(protoPro[i]["value"]["string"].asString());
                    CRYP_INFO<<"mObjectType:"<<static_cast<unsigned int>(protoProps.mObjectType);
                }else if(!protoPro[i]["key"].asString().compare("SlotCapacity")){
                    protoProps.mSlotCapacity = protoPro[i]["value"]["uint32"].asUInt();
                    CRYP_INFO<<"mSlotCapacity:"<<protoProps.mSlotCapacity;
                }else if(!protoPro[i]["key"].asString().compare("Exportability")){
                    protoProps.mExportAllowed = protoPro[i]["value"]["bool"].asBool();
                    CRYP_INFO<<"mExportAllowed:"<<protoProps.mExportAllowed;
                }else if(!protoPro[i]["key"].asString().compare("AllowContentTypeChange")){
                    protoProps.mAllowContentTypeChange = protoPro[i]["value"]["bool"].asBool();
                    CRYP_INFO<<"mAllowContentTypeChange:"<<protoProps.mAllowContentTypeChange;
                }else if(!protoPro[i]["key"].asString().compare("AllocateSpareSlot")){
                    protoProps.mAllocateSpareSlot = protoPro[i]["value"]["bool"].asBool();
                    CRYP_INFO<<"mAllocateSpareSlot:"<<protoProps.mAllocateSpareSlot;
                }else if(!protoPro[i]["key"].asString().compare("ContentAllowedUsage")){
                    protoProps.mContentAllowedUsage = protoPro[i]["value"]["uint32"].asUInt();
                    CRYP_INFO<<"mContentAllowedUsage:"<<protoProps.mContentAllowedUsage;
                }else if(!protoPro[i]["key"].asString().compare("MaxUpdateAllowed")){
                    protoProps.mMaxUpdateAllowed = protoPro[i]["value"]["int32"].asInt();
                    CRYP_INFO<<"mMaxUpdateAllowed:"<<protoProps.mMaxUpdateAllowed;
                }else{

                }

                ret = true;
            }
        }

        if(contentPro && (contentPro.type() == Json::arrayValue)) {
            CRYP_INFO << "contentPro.size:" << contentPro.size();
            for(unsigned int i=0; i<contentPro.size(); i++){
                if(!contentPro[i]["key"].asString().compare("ObjectUid")){

                }else if(!contentPro[i]["key"].asString().compare("AlgId")){
                    contentProps.mAlgId = static_cast<CryptoAlgId>(protoPro[i]["value"]["uint64"].asUInt64());
                    CRYP_INFO<<"mAlgId:"<< contentProps.mAlgId ;
                }else if(!contentPro[i]["key"].asString().compare("ObjectSize")){
                    contentProps.mObjectSize = contentPro[i]["value"]["uint64"].asUInt64();
                    CRYP_INFO<<"mObjectSize:"<<contentProps.mObjectSize;
                }else if(!contentPro[i]["key"].asString().compare("AllowedUsage")){
                    contentProps.mContentAllowedUsage =  contentPro[i]["value"]["uint32"].asUInt();
                    CRYP_INFO<<"CryptoAlgType:"<< contentProps.mContentAllowedUsage;
                }else if(!contentPro[i]["key"].asString().compare("ObjectType")){
                    contentProps.mObjectType =static_cast<CryptoObjectType>(contentPro[i]["value"]["uint32"].asUInt());
                    CRYP_INFO<<"mObjectType:"<< static_cast<std::uint32_t>(contentProps.mObjectType);
                }else{

                }

                ret = true;
            }
        }
        CRYP_INFO<<"protoProps.mslotNum:" <<protoProps.mslotNum<< " contentProps.mObjectSize:"<<contentProps.mObjectSize;

    } catch (std::exception& e) {
        CRYP_ERROR << "Exceptions when parse body of cryptoslot_hz_tsp_pkiProcess.json: " << e.what();
        ifs.close();
        return ret;
    }
    ifs.close();
    return ret;
}
}  // namespace keys
}  // namespace crypto
}  // namespace hozon
}  // namespace neta