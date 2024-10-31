#include <stdio.h>
#include <iostream>
#include <list>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <time.h>
#include "core/vector.h"
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/ocsp.h>
#include "x509_provider.h"
// #include "crypto_server_logger.h"
#include "crypto_provider.h"
#include "signer_private_ctx.h"

#include "crypto_context.h"
#include "base_id_types.h"
#include "hash_function_ctx.h"
#include "crypto_provider.h"
#include "entry_point.h"

using namespace hozon::netaos::crypto;
using namespace hozon::netaos::crypto::cryp;
using namespace hozon::netaos::crypto::x509;
using namespace hozon::netaos::core;
using namespace std;

X509Provider::Uptr provider = LoadX509Provider();

std::shared_ptr<std::vector<uint8_t>> ReadFile(std::string filepath) {

    auto buf = std::make_shared<std::vector<uint8_t>>();
    std::shared_ptr<FILE> file(fopen(filepath.c_str(), "rb"), [](FILE* f){ fclose(f); });
    if (file) {
        size_t size = 0;
        fseek(file.get(), 0, SEEK_END);
        size = ftell(file.get());
        fseek(file.get(), 0, SEEK_SET);
        buf->resize(size);
        if (size != fread(buf->data(), 1, size, file.get())) {
            std::cout << "Read ca.crt failed";
        }
    }

    return buf;
}

void test_SetAsRootOfTrust() {
    string root_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/destination_certs/client.pem";
    auto certdata = ReadFile(root_path);
    auto root = provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(certdata->data()),
            static_cast<std::size_t>(certdata->size())), Serializable::kFormatPemEncoded);
    if (provider) {
        if (provider->SetAsRootOfTrust(root)) {
            cout<< "SetAsRootOfTrust success !"<< endl;
        } else {
            cout<< "SetAsRootOfTrust failed !"<< endl;
        }
       
    } else {
        cout << "X509Provider::getInstance is null"<<endl;
    }
}

void test_ImportCrl() {
    string sourch_path = "/media/cgh/data/qingluan/nos/test/certmanager_client_test/test_certs/sourch_certs/server.crl";
    if (provider) {
        if (provider->ImportCrl(sourch_path)) {
            cout<< "ImportCrl success !"<< endl;
        } else {
            cout<< "ImportCrl failed !"<< endl;
        }
       
    } else {
        cout << "X509Provider::getInstance is null"<<endl;
    }
}

void test_VerifyCert() {
    string server_path = "/home/zhouyuli/下载/test_0817/cert_test/test/server.crt";
    string client_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/sourch_certs/client.pem";
    string root_path = "/home/zhouyuli/下载/test_0817/cert_test/test/root.crt";
    auto certdata = ReadFile(server_path);
    std::cout<< "test_VerifyCert certdata->size()" <<certdata->size()<<std::endl;
    auto server = provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(certdata->data()),
            static_cast<std::size_t>(certdata->size())), Serializable::kFormatPemEncoded);
    if (provider) {
        Certificate::Status server_status = provider->VerifyCert(server, root_path);
        cout << "test_VerifyCert server cert status: "<< static_cast<int>(server_status) <<endl;

        // Certificate::Status client_status = provider->VerifyCert(client_path, root_path);
        // cout << "test_VerifyCert client cert status: "<< static_cast<int>(client_status) <<endl;
       
    } else {
        cout << "test_VerifyCert X509Provider::getInstance is null"<<endl;
    }
   std::cout << "test_VerifyCert finish"<<std::endl;
}

void test_CreateOcspRequest() {
    string cert_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/sourch_certs/ocspCert.pem";
    string issuer_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/sourch_certs/chain.pem";
    string ocspreq_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/destination_certs/ocspRequest.der";
    if (provider) {
        OcspRequest::Uptr uptr = provider->CreateOcspRequest(cert_path, issuer_path);
        if (uptr) {
            if (uptr->ExportASN1OCSPRequest(ocspreq_path)) {
                cout<< "test_CreateOcspRequest success !"<< endl;
            } else {
                cout << "test_CreateOcspRequest ExportASN1OCSPRequest failed!"<<endl;
            }
        } else {
            cout << "test_CreateOcspRequest failed uptr is null!"<<endl;
        }
       
    } else {
        cout << "X509Provider::getInstance is null"<<endl;
    }
}

void test_ParseOcspResponse() {
    string cert_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/destination_certs/ocspResp.der";
    if (provider) {
        OcspResponse::Uptr uptr = provider->ParseOcspResponse(cert_path);
        if (uptr) {
            cout<< "test_ParseOcspResponse success !"<< endl;
            cout<< "test_ParseOcspResponse respStatus : !"<< uptr->RespStatus()<<endl;
        } else {
            cout << "test_ParseOcspResponse failed uptr is null!"<<endl;
        }
    } else {
        cout << "X509Provider::getInstance is null"<<endl;
    }
}

void test_CheckCertStatus() {
    string ocsp_resp = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/destination_certs/ocspResp.der";
    string cert_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/sourch_certs/ocspCert.pem";
    string issuer_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/sourch_certs/ocspIssuer.pem";

    string test_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/sourch_certs/client.pem";
    if (provider) {
        Certificate::Status status1 = provider->CheckCertStatus(cert_path, ocsp_resp, issuer_path);
        cout<< "test_ParseOcspResponse success !"<< endl;
        cout<< "test_ParseOcspResponse cert1 status: !"<<static_cast<int>(status1)<< endl;

        Certificate::Status status2 = provider->CheckCertStatus(test_path, ocsp_resp, issuer_path);
        cout<< "test_ParseOcspResponse success !"<< endl;
        cout<< "test_ParseOcspResponse cert2 status: !"<<static_cast<int>(status2)<< endl;
    } else {
        cout << "X509Provider::getInstance is null"<<endl;
    }
}

void test_ImportCert() {
    {
        string test_path = "/media/cgh/data/qingluan/nos/test/certmanager_client_test/test_certs/sourch_certs/server.crl";
        auto certdata = ReadFile(test_path);
        // std::string cert_path = "/home/zhouyuli/下载/test_0817/cert_test/client.pem";
        auto cert = provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(certdata->data()),
                static_cast<std::size_t>(certdata->size())), Serializable::kFormatPemEncoded);
        // provider->ImportCert(cert, cert_path);
    }
    // {
    //     string test_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/sourch_certs/root.pem";
    //     auto certdata = ReadFile(test_path);
    //     std::string cert_path = "/home/zhouyuli/Downloads/out.pem";
    //     auto cert = provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(certdata->data()),
    //             static_cast<std::size_t>(certdata->size())), Serializable::kFormatPemEncoded);
    //     provider->ImportCert(cert, cert_path);
    // }
    {
        // string test_path = "/home/zhouyuli/WORK/PROJECT/MDC_SOC_ch/netaos/test/certmanager_client_test/test_certs/sourch_certs/client.der";
        // auto certdata = ReadFile(test_path);
        // std::cout<< "certdata->size()" <<certdata->size()<<std::endl;
        // std::string cert_path = "/home/zhouyuli/Downloads/out.der";
        // auto cert = provider->ParseCert(ReadOnlyMemRegion(reinterpret_cast<const unsigned char*>(certdata->data()),
        //         static_cast<std::size_t>(certdata->size())), Serializable::kFormatDerEncoded);
        // provider->ImportCert(cert, cert_path);
    }
}

void test_FindCertByDn() {
    std::string subject_Dn = "/C=CN/ST=CN/O=CN/OU=CN/CN=127.0.0.1/emailAddress=DF";
    std::string issuer_Dn = "/C=CN/ST=CN/L=CN/O=CN/OU=CN/CN=CN/emailAddress=CN";

    auto subjectDn = provider->BuildDn(subject_Dn);
    auto issuerDn = provider->BuildDn(issuer_Dn);

    time_t t = time(NULL);
    auto mycert = provider->FindCertByDn(*subjectDn.get(), *issuerDn.get(), t);

    if (mycert->cert_) {
        X509_NAME* subject_dn = X509_get_subject_name(mycert->cert_);
        char subject_buf[4096];
        X509_NAME_oneline(subject_dn, subject_buf, sizeof(subject_buf));
        printf("test_FindCertByDn curr_cert subject_buf: %s\n", subject_buf);
    } else {
        printf("test_FindCertByDn curr_cert is null \n");
    }
}

void test_CreateCertSignRequest() {
    CryptoProvider::Uptr loadProvider = LoadCryptoProvider();
    auto result_rsaKeyCtx = loadProvider->GeneratePrivateKey(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS,kAllowKdfMaterialAnyUsage,false,true);
    if (result_rsaKeyCtx.HasValue()) {

    } else {
        std::cout << "crypto GeneratePrivateKey failed."<<std::endl;
        std::cout<<result_rsaKeyCtx.Error().Message().data()<<std::endl;
        return;
    }
    auto pk = result_rsaKeyCtx.Value()->GetPublicKey().Value();

    //sign
    auto result_sign_privae_ctx = loadProvider->CreateSignerPrivateCtx(hozon::netaos::crypto::kAlgIdRSA2048SHA384PSS);
    if (result_sign_privae_ctx.HasValue()) {
        result_sign_privae_ctx.Value()->SetKey(*result_rsaKeyCtx.Value().get());
    } else {
        std::cout << "crypto CreateSignerPrivateCtx failed." << std::endl;
        std::cout << result_sign_privae_ctx.Error().Message().data() << std::endl;
        return;
    }


    std::string strDn = "/C=CN/O=jianwangsan/OU=yyl/CN=Test";
    auto dn = provider->BuildDn(strDn);

    std::map<std::uint32_t, std::string> ex_map;

    CertSignRequest::Uptr csr(provider->CreateCertSignRequest(const_cast<SignerPrivateCtx::Uptr&>(result_sign_privae_ctx.Value()), dn, ex_map, 0));
    string csr_path = "/home/zhouyuli/WORK/PROJECT/orin/nos/test/certmanager_client_test/test_certs/destination_certs/hanavivi0823.csr";
    csr->ExportASN1CertSignRequest(csr_path);
}

#include "crypto_adapter.h"

// void test_pki_use_csr() {

//     std::shared_ptr<std::vector<uint8_t>> csr = nullptr;
//     CryptoAdapter cryAdp;
//     std::string ret;
//     crypto::cryp::PrivateKey::Uptrc upri = nullptr;
//     std::cout << "GetCsr called."<<std::endl;

//     // std::string strDn = "/CN="+vin+"_"+sn+"/O=HOZON/OU=EP40_ADCS/C=CN";
//     DnInfo dnInfo;
//     dnInfo.common_name = "CN";
//     dnInfo.organization = "HOZON";
//     dnInfo.organization_unit = "EP40_ADCS";
//     dnInfo.state = "";
//     dnInfo.common_name = "CN";
//     dnInfo.email_address = "";
//     cryAdp.Init("/home/zhouyuli/Downloads/pfx_test/test.json");
//     if(cryAdp.HasKeyInSlot("7890abdde5667c38-231175d3e56b6c37")){
//         upri = std::move(cryAdp.ReadPrivateKey("7890abdde5667c38-231175d3e56b6c37"));
//         csr = cryAdp.CreateClientCsr(*upri.get(),dnInfo);
//         std::cout << "[TSP_PKI:] Has key in slot." << std::endl;
//     }else{
//         std::cout << "[TSP_PKI:]  Do not has key in slot." << std::endl;

//         auto result_rsaKeyCtx = cryAdp.CreateRsaPrivateKey();
//         if (result_rsaKeyCtx.get()) {
//             // upri = std::move(result_rsaKeyCtx.Value());
//             std::cout << "[TSP_PKI:] GeneratePrivateKey finished." << std::endl;
//             csr = cryAdp.CreateClientCsr(*result_rsaKeyCtx.get(), dnInfo);

//         } else {
//             std::cout << "[TSP_PKI:] crypto GeneratePrivateKey failed." << std::endl;
//             return;
//         }
//     }
//     // std::string st(csr->begin(), csr->end());
//     // std::cout << "csr size:"<<csr->size()<<std::endl;
//     // for(unsigned int i = 0;i<csr->size();i++){
//     //     ret[i] = static_cast<unsigned char>(csr->at(i));
//     // }
// }

int main() {
    CryptoServerLogger::GetInstance().setLogLevel(static_cast<int32_t>(
        CryptoServerLogger::CryptoLogLevelType::CRYPTO_INFO));
    CryptoServerLogger::GetInstance().InitLogging(
        "CRYPTO_X509",       // the id of application
        "CRYPTO X509 test",  // the log id of application
        CryptoServerLogger::CryptoLogLevelType::
            CRYPTO_INFO,  // the log
                             // level of
                             // application
        hozon::netaos::log::HZ_LOG2CONSOLE |
          hozon::netaos::log::HZ_LOG2FILE,  // the output log mode
        "./",  // the log file directory, active when output log to file
        10,    // the max number log file , active when output log to file
        20     // the max size of each  log file , active when output log to file
    );
    CryptoServerLogger::GetInstance().CreateLogger("CRYPTO_X509");
    // test_CreateCertSignRequest();
    test_ImportCert();
    // test_Import();
    // test_SetAsRootOfTrust();
    // test_ImportCrl();
    // test_VerifyCert();
    // test_ParseOcspResponse();
    // test_CheckCertStatus();
    // test_FindCertByDn();
    // test_CreateOcspRequest();
    // test_ParseOcspResponse();
    // test_pki_use_csr();
}