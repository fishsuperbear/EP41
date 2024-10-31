#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <memory>
#include <vector>
#include <cstddef>
#include <utility>
#include <fstream>
#include <signal.h>
#include <getopt.h>

// #include "key_slot_content_props.h"
// #include "key_slot_prototype_props.h"

#include "cryp/crypto_provider.h"
#include "keys/key_storage_provider.h"
#include "common/entry_point.h"
// #include "base_id_types.h"
// #include "core/result.h"
// #include "core/instance_specifier.h"
// #include "core/string_view.h"


#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/types.h>
#include <openssl/crypto.h>

#include "json/json.h"
// #include "key_storage_provider.h"
// #include "keyslot.h"
// #include "imp_keyslot.h"
// #include "imp_io_interface.h"
#include "pki_logger.hpp"


static bool stopped_ = false;
/** Signal handler.*/
static void SigHandler(int signum)
{

    std::cout << "Received signal: " << signum  << ". Quitting\n";
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    stopped_ = true;

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
static void SigSetup(void)
{
    struct sigaction action
    {
    };
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);
}

using namespace hozon;
using namespace hozon::netaos::crypto;
using namespace hozon::netaos::crypto::cryp;
using namespace hozon::netaos::crypto::keys;

std::string testJson_path = "/cfg/pki/etc/cryptoslot_hz_tsp_pkiProcess.json";                            

#if 1
int do_crypt(FILE *in, FILE *out, int do_encrypt)
{
    /* Allow enough space in output buffer for additional block */
    unsigned char inbuf[1024],outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    // unsigned char inbuf[] = "0123456789abcdeF";
    int inlen, outlen;
    EVP_CIPHER_CTX *ctx;

    unsigned char key[] = "0123456789abcdeF";
    unsigned char iv[] = "1234567887654321";

    /* Don't set key or IV right away; we want to check lengths */
    ctx = EVP_CIPHER_CTX_new();
    if (!EVP_CipherInit_ex2(ctx, EVP_aes_128_cbc(), NULL, NULL,do_encrypt, NULL)) {

        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    OPENSSL_assert(EVP_CIPHER_CTX_get_key_length(ctx) == 16);
    OPENSSL_assert(EVP_CIPHER_CTX_get_iv_length(ctx) == 16);

    if (!EVP_CipherInit_ex2(ctx, NULL, key, iv, do_encrypt, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }

   while(1) {
        inlen = fread(inbuf, 1, 1024, in);
        if (inlen <= 0){
            std::cout << "inlen:"<<std::dec<<inlen<<std::endl;
            break;
        }
        std::cout << "inlen:"<<inlen<<std::endl;
        if (!EVP_CipherUpdate(ctx, outbuf, &outlen, inbuf, inlen)) {
            EVP_CIPHER_CTX_free(ctx);
            return 0;
        }
        fwrite(outbuf, 1, outlen, out);
    }
    if (!EVP_CipherFinal_ex(ctx, outbuf, &outlen)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    fwrite(outbuf, 1, outlen, out);
    std::cout << "encrytp finish."<<std::endl;
    EVP_CIPHER_CTX_free(ctx);
    return 1;
}

int hashTest(){
    std::string instring = "123456";
    std::vector<std::uint8_t> inbuf(instring.begin(), instring.end());
    std::vector<std::uint8_t> outbuf;
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    auto result = loadProvider->CreateHashFunctionCtx(hozon::netaos::crypto::kAlgIdSHA256);
    if(!result.HasValue()){
        std::cout << "crypto CreateHashFunctionCtx failed."<<std::endl;
        std::cout<<result.Error().Message().data()<<std::endl;
        return -1;
    }

    result.Value()->Start();
    result.Value()->Update(inbuf);
    auto ret = result.Value()->Finish();
    return 0;
}

#endif

#if 1
int SymmetricTest(){
    PKI_INFO << "crypto SymmetricTest begin.";
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    if(!loadProvider.get()){
        std::cout << "crypto LoadCryptoProvider failed."<<std::endl;
        return -1;
    }
    auto result_symmCtx = loadProvider->CreateSymmetricBlockCipherCtx(hozon::netaos::crypto::kAlgIdCBCAES128);
    if(result_symmCtx.HasValue()){
        if(result_symmCtx.Value().get()){

        }else{
            std::cout << "crypto CreateSymmetricBlockCipherCtx failed. result_symmCtx is null" << std::endl; 
            return -1;
        }
    }else{
        std::cout << "crypto CreateSymmetricBlockCipherCtx failed." << std::endl;
        std::cout << result_symmCtx.Error().Message().data() << std::endl;
        return -1;
    }
    PKI_INFO << "crypto CreateSymmetricBlockCipherCtx success.";

    // SymmetricBlockCipherCtx::Uptr symmetricCtx = std::move(result.Value());
    auto gen_sym_key_res = loadProvider->GenerateSymmetricKey(kAlgIdCBCAES128,kAllowDataEncryption|kAllowDataDecryption,true,true);
    if (gen_sym_key_res.HasValue()) {
        if(gen_sym_key_res.Value().get()){

        }else{
            std::cout << "crypto GenerateSymmetricKey failed." << std::endl;
            return -1;
        }
    }else{
        std::cout << "crypto GenerateSymmetricKey failed." << std::endl;
        std::cout << gen_sym_key_res.Error().Message().data() << std::endl;
        return -1;
    }

    PKI_INFO << "crypto GenerateSymmetricKey success.";

    auto sym_key_uptrc = std::move(gen_sym_key_res).Value();
    // SymmetricKey symm_key(kAlgIdCBCAES128,kAllowDataEncryption|kAllowDataDecryption,true,true);
    std::string pliantext = "0123456789abcdeF";
    PKI_INFO<<"pliantext size:"<<pliantext.size();
    ReadOnlyMemRegion testdata(reinterpret_cast<const unsigned char*>(pliantext.data()),pliantext.size());
    PKI_INFO<< "testdata size:"<<testdata.size();
    result_symmCtx.Value()->SetKey(*sym_key_uptrc, CryptoTransform::kEncrypt);
    auto result_encryp = result_symmCtx.Value()->ProcessBlock(testdata, true);
    if(result_encryp.HasValue()){
        netaos::core::Vector<std::uint8_t> temp;
        PKI_INFO<< "symmetric encryptoed data size:"<<result_encryp.Value().size()<<" data:";
        temp.resize(result_encryp.Value().size());
        for(uint32_t i=0;i< result_encryp.Value().size();i++){
            temp[i] = static_cast<std::uint8_t>(result_encryp.Value().at(i));
            std::cout<< result_encryp.Value()[i]<<" ";
        }
        std::cout<<std::endl;
        ReadOnlyMemRegion inbuf(reinterpret_cast<const unsigned char*>(temp.data()),temp.size());
        result_symmCtx.Value()->SetKey(*sym_key_uptrc, CryptoTransform::kDecrypt);
        auto result_decrypto = result_symmCtx.Value()->ProcessBlock(inbuf, true);
        if(result_decrypto.HasValue()){
            std::cout<< "symmetric decryptoed data:";
            for (uint32_t i = 0; i < static_cast<uint32_t>(result_decrypto.Value().size()); i++) {
                std::cout << " " << result_decrypto.Value().at(i);
            }
            std::cout<<std::endl;
        }

    }else{
        PKI_INFO<< "encrypto ProcessBlock failed.";
    }
    return 0;
}
#endif

#if 0
int  genRSAKeyTest(){
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    auto result_rsaKeyCtx = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if(!result_rsaKeyCtx){
        std::cout << "crypto GeneratePrivateKey failed."<<std::endl;
        std::cout<<result_rsaKeyCtx.Error().Message().data()<<std::endl;
        return -1;
    }
    return 0;
}
#endif

#if 1
int  RSAdecryptoTest(){
    // unsigned char plaintxt[] = "abcdefgeosngeon1350450";
    std::string plaintxt("aodngosegnosng");
    std::cout << "crypto RSAdecryptoTest start."<<std::endl;
    ReadOnlyMemRegion indata(reinterpret_cast<uint8_t*>(plaintxt.data()),plaintxt.size());

    std::cout << "crypto RSAdecryptoTest start...."<<std::endl;
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    netaos::core::Vector<std::byte> encryptedata;
    netaos::core::Vector<std::byte> decryptedata;
    auto result_rsaKeyCtx = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if(!result_rsaKeyCtx){
        std::cout << "crypto GeneratePrivateKey failed."<<std::endl;
        std::cout<<result_rsaKeyCtx.Error().Message().data()<<std::endl;
        return -1;
    }

    auto result_rsaEncryptoCtx = loadProvider->CreateEncryptorPublicCtx(kAlgIdRSA2048SHA384PSS);
    if(result_rsaEncryptoCtx.HasValue()){
        // EncryptorPublicCtx::Uptr uptr_encry = std::move(result_rsaEncryptoCtx.Value());
        std::cout << "crypto CreateEncryptorPublicCtx ok."<<std::endl;
    //    PublicKey::Uptrc publickeyPtr = result_rsaKeyCtx.Value()->GetPublicKey().Value();
    //    if(publickeyPtr){
    //         std::cout << "publickey addr:"<< publickeyPtr.get()<< std::endl;
    //    }
        // result_rsaEncryptoCtx.Value()->SetKey(*publickeyPtr.get());
        auto ret_pub = result_rsaKeyCtx.Value()->GetPublicKey();
        if(ret_pub.HasValue()){
            // std::cout<<"ret_pub name:"<<ret_pub.Value()->get_myName()<<std::endl;
            result_rsaEncryptoCtx.Value()->SetKey(*ret_pub.Value().get());
        }

        std::cout << "crypto CreateEncryptorPublicCtx SetKey ok."<<std::endl;

        auto result_encry = result_rsaEncryptoCtx.Value()->ProcessBlock(indata, true);
        if(result_encry.HasValue()){
            std::cout<<" encrypte data:"<<std::endl;
            for(size_t i=0;i<result_encry.Value().size();i++){
                std::cout <<std::hex<< static_cast<char>(result_encry.Value().at(i));
            }
            std::cout<<std::endl;
        }
        encryptedata.reserve(result_encry.Value().size());
        for (const auto& elem : result_encry.Value()) {
            encryptedata.emplace_back(static_cast<std::byte>(elem)); // 将 uint8_t 转换为 std::byte 并赋值给 outputVec
        }
        // encryptedata(result_encry.Value().begin(), result_encry.Value().end());
    } else {
        std::cout << "error  no crypto CreateEncryptorPublicCtx ."<<std::endl;
    }

    std::cout << "crypto CreateEncryptorPublicCtx SetKey over."<<std::endl;

    ReadOnlyMemRegion inencrydata(reinterpret_cast<uint8_t*>(encryptedata.data()),encryptedata.size());
    auto result_rsaDecryptoCtx = loadProvider->CreateDecryptorPrivateCtx(kAlgIdRSA2048SHA384PSS);
    if(result_rsaDecryptoCtx.HasValue()){
        // EncryptorPublicCtx::Uptr uptr_encry = std::move(result_rsaEncryptoCtx.Value());
        result_rsaDecryptoCtx.Value()->SetKey(*result_rsaKeyCtx.Value().get());
        auto result_decry = result_rsaDecryptoCtx.Value()->ProcessBlock(inencrydata, true);
        if(result_decry.HasValue()){
            std::cout<<"decrypte data:"<<std::endl;
            for(size_t i=0;i<result_decry.Value().size();i++){
                std::cout << static_cast<char>(result_decry.Value().at(i));
            }
            std::cout<<std::endl;
        } else {
            std::cout<<"error no decrypte data:"<<std::endl;
        }
    }


    return 0;
}
#endif

#if 1
int sign_verifyTest(){
    std::string instring = "123456";
    ReadOnlyMemRegion indata(reinterpret_cast<uint8_t*>(instring.data()),instring.size());
    ReadOnlyMemRegion context;
    std::vector<std::uint8_t> inbuf(instring.begin(), instring.end());
    std::vector<std::uint8_t> outbuf;
    netaos::core::Vector<std::byte> signValue;
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    auto result_rsaKeyCtx = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if(result_rsaKeyCtx.HasValue()){

    }else{
        std::cout << "crypto GeneratePrivateKey failed."<<std::endl;
        std::cout<<result_rsaKeyCtx.Error().Message().data()<<std::endl;
        return -1;
    }

    //sign
    auto result_sign_privae_ctx = loadProvider->CreateSignerPrivateCtx(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS);
    if (result_sign_privae_ctx.HasValue()) {
        result_sign_privae_ctx.Value()->SetKey(*result_rsaKeyCtx.Value().get());
        auto result_sign = result_sign_privae_ctx.Value()->Sign(indata,context);
        if(result_sign.HasValue()){
            // signValue = result_sign.Value();
            signValue.reserve(result_sign.Value().size());
            for (const auto& elem : result_sign.Value()) {
                signValue.emplace_back(static_cast<std::byte>(elem)); // 将 uint8_t 转换为 std::byte 并赋值给 outputVec
            }
        }else{
            std::cout << "crypto Sign error." << std::endl;
        }
        std::cout << "crypto Sign finish." << std::endl;

    }else{
        std::cout << "crypto CreateSignerPrivateCtx failed." << std::endl;
        std::cout << result_sign_privae_ctx.Error().Message().data() << std::endl;
        return -1;
    }


    //veviry
    if(signValue.empty()){
        std::cout << "signValue is empty,do not verify." << std::endl;
    }

    ReadOnlyMemRegion signature(reinterpret_cast<uint8_t*>(signValue.data()),signValue.size());

    auto result_verify_public_ctx = loadProvider->CreateVerifierPublicCtx(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS);
    if(result_verify_public_ctx.HasValue()){
        result_verify_public_ctx.Value()->SetKey(*result_rsaKeyCtx.Value()->GetPublicKey().Value().get());
        std::cout << "result_verify_public_ctx.Value()->SetKey finish." << std::endl;
        result_verify_public_ctx.Value()->Verify(indata,signature,context);
    }


    
    return 0;
}
#endif

#if 1
void keys_saveKeysTest() {
    // KeySlot::Uptr keyslot_uptr = nullptr;
    // IOInterface::Uptr ioInter_uptr = nullptr;
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    KeyStorageProvider::Uptr keysLoadProvider = LoadKeyStorageProvider();

    //////////////////////////////////////////////////////////////////////////
    // generate key0. 1
    auto rsa_key = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if(!rsa_key){
        std::cout << "crypto GeneratePrivateKey failed."<<std::endl;
        std::cout<<rsa_key.Error().Message().data()<<std::endl;
        return;
    }

    std::string temp = testJson_path + "_" + "7890abdde5667c38-231175d3e56b6c37";

    netaos::core::StringView sv(temp.data());
    netaos::core::InstanceSpecifier iSpec(sv);

    // get the keySlot though the .json file.2
    auto result_slot = keysLoadProvider->LoadKeySlot(iSpec);
    const KeySlot::Uptr& keyslot_uptr = result_slot.Value();
    
    // generate the ioInterface base on the slot.3
    auto result_io =  keyslot_uptr->Open(false,false);
    const IOInterface::Uptr& ioInter_upr = result_io.Value();

    rsa_key.Value().get()->Save(*(ioInter_upr.get()));

    // save the content of key to ioInterface.4
    keyslot_uptr->SaveCopy(*(ioInter_upr.get()));

    // push back slot to vector
    TransactionScope targetSlots;
    targetSlots.push_back(keyslot_uptr.get());

    //////////////////////////////////////////////////////////////////////////
    //generate key1
    auto rsa_key1 = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if(!rsa_key1){
        std::cout << "crypto GeneratePrivateKey failed.1"<<std::endl;
        std::cout<<rsa_key1.Error().Message().data()<<std::endl;
        return;
    }
    temp = testJson_path + "_" + "4560abddecd67234-567175d5d56b6a45";
    netaos::core::StringView sv1(temp.data());
    netaos::core::InstanceSpecifier iSpec1(sv1);

    auto result_slot1 = keysLoadProvider->LoadKeySlot(iSpec1);
    const KeySlot::Uptr& keyslot_uptr1 = result_slot1.Value();
    auto result_io1 =  keyslot_uptr1->Open(false,false);
    const IOInterface::Uptr& ioInter_uptr1 = result_io1.Value();
    rsa_key1.Value().get()->Save(*(ioInter_uptr1.get()));
    keyslot_uptr1->SaveCopy(*(ioInter_uptr1.get()));
    // push back slot to vector
    targetSlots.push_back((keyslot_uptr1.get()));

    // update transaction
    TransactionId id = (keysLoadProvider->BeginTransaction(targetSlots)).Value();
    std::cout <<"TransactionId:"<<id<<std::endl;
    keysLoadProvider->CommitTransaction(id);

    // for test one more transaction
    // TransactionId id1 = (keyLoadProvide->BeginTransaction(targetSlots)).Value();
    // std::cout <<"TransactionId id1: "<<id1<<std::endl;
    // keyLoadProvide->CommitTransaction(id1);
}
#endif
int do_FileCrypt(const std::string& inPath, const std::string& outPath, int do_encrypt)
{
    FILE* in = fopen(inPath.c_str(), "rb");
    FILE* out = fopen(outPath.c_str(), "wb");
    /* Allow enough space in output buffer for additional block */
    unsigned char inbuf[1024],outbuf[1024 + EVP_MAX_BLOCK_LENGTH];
    // unsigned char inbuf[] = "0123456789abcdeF";
    int inlen, outlen;
    EVP_CIPHER_CTX *ctx;

    unsigned char key[] = "0123456789abcdeF";
    unsigned char iv[] = "1234567887654321";

    /* Don't set key or IV right away; we want to check lengths */
    ctx = EVP_CIPHER_CTX_new();
    if (!EVP_CipherInit_ex2(ctx, EVP_aes_128_cbc(), NULL, NULL,do_encrypt, NULL)) {

        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    OPENSSL_assert(EVP_CIPHER_CTX_get_key_length(ctx) == 16);
    OPENSSL_assert(EVP_CIPHER_CTX_get_iv_length(ctx) == 16);

    if (!EVP_CipherInit_ex2(ctx, NULL, key, iv, do_encrypt, NULL)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }

   while(1) {
        inlen = fread(inbuf, 1, 1024, in);
        if (inlen <= 0){
            std::cout << "inlen:"<<std::dec<<inlen<<std::endl;
            break;
        }
        std::cout << "inlen:"<<inlen<<std::endl;
        if (!EVP_CipherUpdate(ctx, outbuf, &outlen, inbuf, inlen)) {
            EVP_CIPHER_CTX_free(ctx);
            return 0;
        }
        fwrite(outbuf, 1, outlen, out);
    }
    if (!EVP_CipherFinal_ex(ctx, outbuf, &outlen)) {
        EVP_CIPHER_CTX_free(ctx);
        return 0;
    }
    fwrite(outbuf, 1, outlen, out);
    fclose(in);
    fclose(out);
    std::cout << "encrytp finish."<<std::endl;
    EVP_CIPHER_CTX_free(ctx);
    return 1;
}

void test_crypt() {
    std::string in_path = "/cfg/pki/keys/3-7890abdde5667c38-231175d3e56b6c37.json";
    std::string out_path = "/cfg/pki/keys/1.json";
    std::string out1_path = "/cfg/pki/keys/2.json";
    do_FileCrypt(in_path, out_path, 1);
    do_FileCrypt(out_path, out1_path, 0);
}

void X509_ParseCert_test() {
    std::string cert = "/cfg/pki/certs/device.pem";
    std::vector<std::uint8_t> pay;
    std::ios_base::openmode mode = std::ios::binary|std::ios::in;
    std::fstream fs(cert,mode);
     if(!fs.is_open()){
        PKI_ERROR<<"fs open failed, cert:"<<cert;
    }else{
        fs.seekg(0, std::ios::end);
        uint64_t length = fs.tellg();
        fs.seekg(0);
        pay.resize(length);
        fs.read(reinterpret_cast<char*>(pay.data()),length);
        fs.close();
    }
    
    PKI_INFO<< "LoadX509Provider begin.";
    x509::X509Provider::Uptr x509_provider = LoadX509Provider();
    PKI_INFO<< "LoadX509Provider finish.";


    auto ret_cert = x509_provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(pay.data()),
            static_cast<std::size_t>(pay.size())), Serializable::kFormatPemEncoded);
    PKI_INFO <<"ret_cert starttime:" << ret_cert->StartTime();
    PKI_INFO<< "X509_ParseCert test finish";
}

int main(int argc, char* argv[])
{
	SigSetup();

    PKILogger::GetInstance().setLogLevel(static_cast<int32_t>(PKILogger::CryptoLogLevelType::CRYPTO_TRACE));
    PKILogger::GetInstance().InitLogging("crypto_pki", "crypto_pki",
                                                PKILogger::CryptoLogLevelType::CRYPTO_TRACE,                  //the log level of application
                                                hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE,  //the output log mode
                                                "/opt/usr/log/soc_log/",                                                                  //the log file directory, active when output log to file
                                                10,                                                                    //the max number log file , active when output log to file
                                                20                                                                     //the max size of each  log file , active when output log to file
    );
    PKILogger::GetInstance().CreateLogger("crypto_pki");
    PKI_INFO<< "Crypto_PKILog init finish.";

    // keys_saveKeysTest();

    // const struct option long_options[] = {
    //     {"usecase", required_argument, nullptr, 'u'},
    //     {nullptr, 0, nullptr, 0}
    // };
    // CryptoCmClient::Instance().Init();
    // SymmetricTest();
    // RSAdecryptoTest();
    // int usecase =0xff;
    // bool write_to_file = false;
    // while (1) {
        // int option_index = 0;
        // int c = getopt_long(argc, argv, "u:", long_options, &option_index);

        // bool parse_end = false;
        // switch (c) {
        //     case 's':
        //         usecase = std::stoul(optarg);
        //     break;
        //     default:
        //         std::cout << argv[1] << " -u <usecase>" << std::endl;
        //     break;
        // }

        // if (parse_end) {
        //     break;
        // } 

    // }

    // switch (usecase) {
    // case 1:
    //     SymmetricTest();
    // break;
    // default:
    // std::cout << "Unsupported usecase.\n";
    // break;
    // }

    
    // KeySlotPrototypeProps protoProps;
    // KeySlotContentProps contentProps;
    // std::string uuid("7890abdde5667c38-231175d3e56b6c37");
    // // bool ret = false;

    // parseJson(uuid, protoProps, contentProps);
    // unsigned char inbuf[] = "0123456789abcdeF";

    // BaseA* bb = new childB;
    // childB& cc = dynamic_cast<childB&>(*bb);

    // hashTest();
    // FILE *input = fopen("input", "r");
    // FILE *output = fopen("output", "wb");
    // do_crypt(input,output,1);
    // fclose(input);
    // fclose(output);
    // symmeticTest();
    // genRSAKeyTest();
    // RSAdecryptoTest();
    // int i ;
    // const int &j = i;
    // sign_verifyTest();
    std::string in = "/preset/sec/oem_keys_preset.yaml";
    std::string out = "/preset/sec/oem_keys_preset.yaml.encrypted";
    do_FileCrypt(in, out, 1);


    do_FileCrypt("/home/zhouyuli/Downloads/oem_keys_preset.yaml.encrypted", "/home/zhouyuli/Downloads/oem_keys_preset.yaml", 0);
    // SymmetricTest();
    std::cout << "crypto test over."<< std::endl;

	while (!stopped_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

	return 0;
}