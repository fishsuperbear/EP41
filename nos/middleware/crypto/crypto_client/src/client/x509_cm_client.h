#pragma once

#include <atomic>
#include <vector>
#include "common/inner_types.h"
#include "cm/include/method.h"
#include "idl/generated/crypto.h"
#include "idl/generated/cryptoPubSubTypes.h"
#include "cryp/cryobj/crypto_object.h"
#include "x509/cimp_x509_dn.h"
#include "common/crypto_logger.hpp"
#include "x509/certificate.h"


namespace hozon {
namespace netaos {
namespace crypto {
namespace x509{

class X509CmClient {

public:
    using ParseCertMethod = hozon::netaos::cm::Client<ParseCertRequest, ParseCertResult>;
    using CreateCertSignMethod = hozon::netaos::cm::Client<CreateCertSignRequest, CreateCSRResult>;
    using ExportASN1CertSignMethod = hozon::netaos::cm::Client<ExportASN1CertSignRequest, ExportASN1CertSignResult>;
    using ImportCertMethod = hozon::netaos::cm::Client<ImportCertRequest, ImportCertResult>;
    using VerifyCertMethod = hozon::netaos::cm::Client<VerifyCertRequest, VerifyCertResult>;
    using FindCertByDnMethod = hozon::netaos::cm::Client<FindCertByDnRequest, FindCertByDnResult>;
    using BuildDnMethod = hozon::netaos::cm::Client<BuildDnRequest, BuildDnResult>;

    using SetDnMethod = hozon::netaos::cm::Client<SetDnRequest, SetDnResult>;
    using GetDnStringMethod = hozon::netaos::cm::Client<GetDnStringRequest, DnCommonResult>;
    using GetAttributeMethod = hozon::netaos::cm::Client<GetAttributeRequest, DnCommonResult>;
    using GetAttributeWithIndexMethod = hozon::netaos::cm::Client<GetAttributeWithIndexRequest, DnCommonResult>;
    using SetAttributeMethod = hozon::netaos::cm::Client<SetAttributeRequest, SetAttributeResult>;

    using GetPathLimitMethod = hozon::netaos::cm::Client<GetPathLimitRequest, GetPathLimitResult>;
    using IsCaMethod = hozon::netaos::cm::Client<IsCaRequest, CertCommonResult>;
    using SubjectDnMethod = hozon::netaos::cm::Client<SubjectDnRequest, SubjectDnResult>;
    using AuthorityKeyIdMethod = hozon::netaos::cm::Client<AuthorityKeyIdRequest, AuthorityKeyIdResult>;
    using StartTimeMethod = hozon::netaos::cm::Client<StartTimeRequest, StartTimeResult>;
    using EndTimeMethod = hozon::netaos::cm::Client<EndTimeRequest, EndTimeResult>;
    using GetStatusMethod = hozon::netaos::cm::Client<GetStatusRequest, GetStatusResult>;
    using IsRootMethod = hozon::netaos::cm::Client<IsRootRequest, CertCommonResult>;
    using IssuerDnMethod = hozon::netaos::cm::Client<IssuerDnRequest, IssuerDnResult>;
    using SerialNumberMethod = hozon::netaos::cm::Client<SerialNumberRequest, SerialNumberResult>;
    using SubjectKeyIdMethod = hozon::netaos::cm::Client<SubjectKeyIdRequest, SubjectKeyIdResult>;
    using VerifyMeMethod = hozon::netaos::cm::Client<VerifyMeRequest, CertCommonResult>;
    using X509VersionMethod = hozon::netaos::cm::Client<X509VersionRequest, X509VersionResult>;

    static X509CmClient& Instance();
    static void Destroy();

    bool Init();
    void Deinit();
    void Stop();
    int32_t CheckInit(){
        if(!inited_){
            return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
        }else{
            return static_cast<int32_t>(CryptoErrc::kSuccess);
        }
    }

    int32_t CheckCommonResult(int cm_res,std::string function_name){
        if (cm_res != 0) {
            CRYP_INFO << function_name<<" failed.";
            return static_cast<int32_t>(CryptoErrc::kCommunicationError);
        } else {
            CRYP_INFO << function_name<<" success.";
            return 0;
        }
    }

    // int32_t GenerateKey(int32_t key_type, uint64_t alg_id, uint32_t allowed_usage, bool is_session, bool is_exportable, CryptoKeyRef& key_ref);
    int32_t ParseCert(const std::vector<uint8_t>& certMem, const uint32_t formatId,CryptoCertRef& cert_ref);
    int32_t CreateCSR(CipherCtxRef signerCtx, X509DNRef& derSubjectDN, std::map<std::uint32_t,
        std::string>& x509Extensions, std::uint32_t version, CmX509_Ref&  csr);
    int32_t ExportCSR(CmX509_Ref&  csr_ref, std::vector<uint8_t>& csr_vec);
    bool ImportCert(CryptoCertRef& cert_ref, const std::string destCertPath = "");
    CertStatus VerifyCert(CryptoCertRef& cert_ref,const std::string rootCertPath);
    int32_t FindCertByDn(const X509DN &subjectDn,const X509DN &issuerDn, time_t validityTimePoint,CryptoCertRef& cert_ref) noexcept;
    int32_t BuildDn(std::string dn,X509DNRef& dn_ref) noexcept;

    bool SetDn (const std::string dn,X509DNRef& dn_ref);
    std::string GetDnString (X509DNRef& dn_ref);
    std::string GetAttribute (const X509DN::AttributeId& id,X509DNRef& dn_ref);
    std::string GetAttributeWithIndex (const X509DN::AttributeId& id, const uint32_t& index,const X509DNRef& dn_ref);
    bool SetAttribute (const X509DN::AttributeId& id,const std::string& attribute,X509DNRef& dn_ref);


    uint32_t GetPathLimit(CryptoCertRef& cert_ref);
    bool IsCa(CryptoCertRef& cert_ref);
    X509DNRef SubjectDn(CryptoCertRef& cert_ref);
    std::string AuthorityKeyId(CryptoCertRef& cert_ref);
    time_t StartTime(CryptoCertRef& cert_ref);
    time_t EndTime(CryptoCertRef& cert_ref);
    Certificate::Status GetStatus(CryptoCertRef& cert_ref);
    bool IsRoot(CryptoCertRef& cert_ref);
    X509DNRef IssuerDn(CryptoCertRef& cert_ref);
    std::string SerialNumber(CryptoCertRef& cert_ref);
    std::string SubjectKeyId(CryptoCertRef& cert_ref);
    bool VerifyMe(CryptoCertRef& caCert);
    std::uint32_t X509Version(CryptoCertRef& cert_ref);
    
private:
    X509CmClient();
    ~X509CmClient();

    std::atomic<bool> inited_{false};
    std::atomic<bool> stopped_{false};
    std::unique_ptr<ParseCertMethod> parse_cert_ = nullptr;
    std::unique_ptr<CreateCertSignMethod> create_cert_sign_request_ = nullptr;
    std::unique_ptr<ExportASN1CertSignMethod> export_asn1_cert_sign_request_ = nullptr;
    std::unique_ptr<ImportCertMethod> import_cert_ = nullptr;
    std::unique_ptr<VerifyCertMethod> verify_cert_ = nullptr;
    std::unique_ptr<FindCertByDnMethod> find_cert_by_dn_ = nullptr;
    std::unique_ptr<BuildDnMethod> build_dn_ = nullptr;

    std::unique_ptr<SetDnMethod> set_dn_ = nullptr;
    std::unique_ptr<GetDnStringMethod> get_dn_string_ = nullptr;
    std::unique_ptr<GetAttributeMethod> get_dn_attribute_ = nullptr;
    std::unique_ptr<GetAttributeWithIndexMethod> get_dn_attribut_with_index_ = nullptr;
    std::unique_ptr<SetAttributeMethod> set_dn_attribut_ = nullptr;

    std::unique_ptr<GetPathLimitMethod> get_path_limit_ = nullptr;
    std::unique_ptr<IsCaMethod> is_ca_ = nullptr;
    std::unique_ptr<SubjectDnMethod> subject_dn_ = nullptr;
    std::unique_ptr<AuthorityKeyIdMethod> authority_key_id_ = nullptr;
    std::unique_ptr<StartTimeMethod> start_time_ = nullptr;
    std::unique_ptr<EndTimeMethod> end_time_ = nullptr;
    std::unique_ptr<GetStatusMethod> get_status_ = nullptr;
    std::unique_ptr<IsRootMethod> is_root_ = nullptr;
    std::unique_ptr<IssuerDnMethod> issuer_dn_ = nullptr;
    std::unique_ptr<SerialNumberMethod> serial_num_ = nullptr;
    std::unique_ptr<SubjectKeyIdMethod> subject_key_id_ = nullptr;
    std::unique_ptr<VerifyMeMethod> verify_me_ = nullptr;
    std::unique_ptr<X509VersionMethod> x509_version_ = nullptr;

};

}
}
}
}