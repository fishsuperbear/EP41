#include "x509_cm_client.h"
#include <mutex>
#include <memory>
#include "cm/include/method.h"
#include "common/crypto_error_domain.h"
#include "common/crypto_logger.hpp"
#include "common/type_converter.h"
#include "crypto_cm_client.h"
#include "x509/certificate.h"
#include "x509/cimp_x509_dn.h"

namespace hozon {
namespace netaos {
namespace crypto {
namespace x509{

// extern  const uint32_t CRYPTO_DOMAIN;
static std::mutex sinstance_mutex_;
static X509CmClient* sinstance_ = nullptr;
X509CmClient& X509CmClient::Instance()  {
    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (!sinstance_) {
        sinstance_ = new X509CmClient();
    }
    return *sinstance_;
}

void X509CmClient::Destroy()  {
    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (sinstance_) {
        delete sinstance_;
    }
}

bool X509CmClient::Init()  {
    // parse_cert_.reset(new ParseCertMethod(std::make_shared<ParseCertRequestPubSubType>(), std::make_shared<ParseCertResultPubSubType>()));
    // parse_cert_->Init(CRYPTO_DOMAIN, "ParseCertRequest");
    // while (!stopped_) {
    //     if (0 == parse_cert_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:ParseCert Method";
    //         break;
    //     }
    // }

    create_cert_sign_request_.reset(new CreateCertSignMethod(std::make_shared<CreateCertSignRequestPubSubType>(), std::make_shared<CreateCSRResultPubSubType>()));
    create_cert_sign_request_->Init(CRYPTO_DOMAIN, "CreateCertSignRequest");
    while (!stopped_) {
        if (0 == create_cert_sign_request_->WaitServiceOnline(500)) {
            CRYP_INFO<<"find service:CreateCertSign Method";
            break;
        }
    }

    export_asn1_cert_sign_request_.reset(new ExportASN1CertSignMethod(std::make_shared<ExportASN1CertSignRequestPubSubType>(), std::make_shared<ExportASN1CertSignResultPubSubType>()));
    export_asn1_cert_sign_request_->Init(CRYPTO_DOMAIN, "ExportASN1CertSignRequest");
    while (!stopped_) {
        if (0 == export_asn1_cert_sign_request_->WaitServiceOnline(500)) {
            CRYP_INFO<<"find service:ExportASN1CertSign Method";
            break;
        }
    }

    // import_cert_.reset(new ImportCertMethod(std::make_shared<ImportCertRequestPubSubType>(), std::make_shared<ImportCertResultPubSubType>()));
    // import_cert_->Init(CRYPTO_DOMAIN, "CryptoRequest");
    // while (!stopped_) {
    //     if (0 == import_cert_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:ImportCert Method";
    //         break;
    //     }
    // }

    // verify_cert_.reset(new VerifyCertMethod(std::make_shared<VerifyCertRequestPubSubType>(), std::make_shared<VerifyCertResultPubSubType>()));
    // verify_cert_->Init(CRYPTO_DOMAIN, "VerifyCertRequest");
    // while (!stopped_) {
    //     if (0 == verify_cert_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:VerifyCert Method";
    //         break;
    //     }
    // }

    // find_cert_by_dn_.reset(new FindCertByDnMethod(std::make_shared<FindCertByDnRequestPubSubType>(), std::make_shared<FindCertByDnResultPubSubType>()));
    // find_cert_by_dn_->Init(CRYPTO_DOMAIN, "FindCertByDnRequest");
    // while (!stopped_) {
    //     if (0 == find_cert_by_dn_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:FindCertByDn Method";
    //         break;
    //     }
    // }


    build_dn_.reset(new BuildDnMethod(std::make_shared<BuildDnRequestPubSubType>(), std::make_shared<BuildDnResultPubSubType>()));
    build_dn_->Init(CRYPTO_DOMAIN, "BuildDnRequest");
    while (!stopped_) {
        if (0 == build_dn_->WaitServiceOnline(500)) {
            CRYP_INFO<<"find service:BuildDn Method";
            break;
        }
    }

    set_dn_.reset(new SetDnMethod(std::make_shared<SetDnRequestPubSubType>(), std::make_shared<SetDnResultPubSubType>()));
    set_dn_->Init(CRYPTO_DOMAIN, "SetDnRequest");
    while (!stopped_) {
        if (0 == set_dn_->WaitServiceOnline(500)) {
            CRYP_INFO<<"find service:SetDn Method";
            break;
        }
    }

    // get_dn_string_.reset(new GetDnStringMethod(std::make_shared<GetDnStringRequestPubSubType>(), std::make_shared<DnCommonResultPubSubType>()));
    // get_dn_string_->Init(CRYPTO_DOMAIN, "GetDnStringRequest");
    // while (!stopped_) {
    //     if (0 == get_dn_string_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetDnString Method";
    //         break;
    //     }
    // }

    // get_dn_attribute_.reset(new GetAttributeMethod(std::make_shared<GetAttributeRequestPubSubType>(), std::make_shared<DnCommonResultPubSubType>()));
    // get_dn_attribute_->Init(CRYPTO_DOMAIN, "GetAttributeRequest");
    // while (!stopped_) {
    //     if (0 == get_dn_attribute_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetAttribute Method";
    //         break;
    //     }
    // }

    // get_dn_attribut_with_index_.reset(new GetAttributeWithIndexMethod(std::make_shared<GetAttributeWithIndexRequestPubSubType>(), std::make_shared<DnCommonResultPubSubType>()));
    // get_dn_attribut_with_index_->Init(CRYPTO_DOMAIN, "GetAttributeWithIndexRequest");
    // while (!stopped_) {
    //     if (0 == get_dn_attribut_with_index_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetAttributeWithIndex Method";
    //         break;
    //     }
    // }

    // set_dn_attribut_.reset(new SetAttributeMethod(std::make_shared<SetAttributeRequestPubSubType>(), std::make_shared<SetAttributeResultPubSubType>()));
    // set_dn_attribut_->Init(CRYPTO_DOMAIN, "GetAttributeWithIndexRequest");
    // while (!stopped_) {
    //     if (0 == set_dn_attribut_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetAttributeWithIndex Method";
    //         break;
    //     }
    // }


    // get_path_limit_.reset(new GetPathLimitMethod(std::make_shared<GetPathLimitRequestPubSubType>(), std::make_shared<GetPathLimitResultPubSubType>()));
    // get_path_limit_->Init(CRYPTO_DOMAIN, "GetPathLimitRequest");
    // while (!stopped_) {
    //     if (0 == get_path_limit_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetPathLimitRequest Method";
    //         break;
    //     }
    // }

    // is_ca_.reset(new IsCaMethod(std::make_shared<IsCaRequestPubSubType>(), std::make_shared<CertCommonResultPubSubType>()));
    // is_ca_->Init(CRYPTO_DOMAIN, "IsCaRequest");
    // while (!stopped_) {
    //     if (0 == is_ca_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:IsCaRequest Method";
    //         break;
    //     }
    // }

    // subject_dn_.reset(new SubjectDnMethod(std::make_shared<SubjectDnRequestPubSubType>(), std::make_shared<SubjectDnResultPubSubType>()));
    // subject_dn_->Init(CRYPTO_DOMAIN, "SubjectDnRequest");
    // while (!stopped_) {
    //     if (0 == subject_dn_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:SubjectDnRequest Method";
    //         break;
    //     }
    // }

    // authority_key_id_.reset(new AuthorityKeyIdMethod(std::make_shared<AuthorityKeyIdRequestPubSubType>(), std::make_shared<AuthorityKeyIdResultPubSubType>()));
    // authority_key_id_->Init(CRYPTO_DOMAIN, "AuthorityKeyIdRequest");
    // while (!stopped_) {
    //     if (0 == authority_key_id_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:AuthorityKeyIdRequest Method";
    //         break;
    //     }
    // }

    // start_time_.reset(new StartTimeMethod(std::make_shared<StartTimeRequestPubSubType>(), std::make_shared<StartTimeResultPubSubType>()));
    // start_time_->Init(CRYPTO_DOMAIN, "StartTimeRequest");
    // while (!stopped_) {
    //     if (0 == start_time_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:StartTimeRequest Method";
    //         break;
    //     }
    // }

    // end_time_.reset(new EndTimeMethod(std::make_shared<EndTimeRequestPubSubType>(), std::make_shared<EndTimeResultPubSubType>()));
    // end_time_->Init(CRYPTO_DOMAIN, "EndTimeRequest");
    // while (!stopped_) {
    //     if (0 == end_time_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:EndTimeMethod Method";
    //         break;
    //     }
    // }

    // get_status_.reset(new GetStatusMethod(std::make_shared<GetStatusRequestPubSubType>(), std::make_shared<GetStatusResultPubSubType>()));
    // get_status_->Init(CRYPTO_DOMAIN, "GetStatusRequest");
    // while (!stopped_) {
    //     if (0 == get_status_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:GetStatusRequest Method";
    //         break;
    //     }
    // }

    // is_root_.reset(new IsRootMethod(std::make_shared<IsRootRequestPubSubType>(), std::make_shared<CertCommonResultPubSubType>()));
    // is_root_->Init(CRYPTO_DOMAIN, "IsRootRequest");
    // while (!stopped_) {
    //     if (0 == is_root_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:IsRootRequest Method";
    //         break;
    //     }
    // }

    // issuer_dn_.reset(new IssuerDnMethod(std::make_shared<IssuerDnRequestPubSubType>(), std::make_shared<IssuerDnResultPubSubType>()));
    // issuer_dn_->Init(CRYPTO_DOMAIN, "IssuerDnRequest");
    // while (!stopped_) {
    //     if (0 == issuer_dn_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:IssuerDnRequest Method";
    //         break;
    //     }
    // }

    // serial_num_.reset(new SerialNumberMethod(std::make_shared<SerialNumberRequestPubSubType>(), std::make_shared<SerialNumberResultPubSubType>()));
    // serial_num_->Init(CRYPTO_DOMAIN, "SerialNumberRequest");
    // while (!stopped_) {
    //     if (0 == serial_num_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:SerialNumberRequest Method";
    //         break;
    //     }
    // }

    // subject_key_id_.reset(new SubjectKeyIdMethod(std::make_shared<SubjectKeyIdRequestPubSubType>(), std::make_shared<SubjectKeyIdResultPubSubType>()));
    // subject_key_id_->Init(CRYPTO_DOMAIN, "SubjectKeyIdRequest");
    // while (!stopped_) {
    //     if (0 == subject_key_id_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:SubjectKeyIdRequest Method";
    //         break;
    //     }
    // }

    // verify_me_.reset(new VerifyMeMethod(std::make_shared<VerifyMeRequestPubSubType>(), std::make_shared<CertCommonResultPubSubType>()));
    // verify_me_->Init(CRYPTO_DOMAIN, "VerifyMeRequest");
    // while (!stopped_) {
    //     if (0 == verify_me_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:VerifyMeRequest Method";
    //         break;
    //     }
    // }

    // x509_version_.reset(new X509VersionMethod(std::make_shared<X509VersionRequestPubSubType>(), std::make_shared<X509VersionResultPubSubType>()));
    // x509_version_->Init(CRYPTO_DOMAIN, "X509VersionRequest");
    // while (!stopped_) {
    //     if (0 == x509_version_->WaitServiceOnline(500)) {
    //         CRYP_INFO<<"find service:X509VersionRequest Method";
    //         break;
    //     }
    // }

    // auto release_object_req_ps_type = std::make_shared<ReleaseObjectRequestPubSubType>();
    // auto release_object_res_ps_type = std::make_shared<ReleaseObjectResultPubSubType>();
    // release_object_method_.reset(new ReleaseObjectMethod(release_object_req_ps_type, release_object_res_ps_type));
    // release_object_method_->Init(CRYPTO_DOMAIN, "ReleaseObjectRequest");
    // std::cout<<"find service:ReleaseObjectMethod"<<std::endl;

    inited_ = true;
    return inited_;
}

void X509CmClient::Deinit()  {
    if(parse_cert_.get()) parse_cert_->Deinit();
    if(create_cert_sign_request_.get()) create_cert_sign_request_->Deinit();
    if(export_asn1_cert_sign_request_.get()) export_asn1_cert_sign_request_->Deinit();
    if(import_cert_.get()) import_cert_->Deinit();
    if(verify_cert_.get()) verify_cert_->Deinit();
    if(find_cert_by_dn_.get()) find_cert_by_dn_->Deinit();
    if(build_dn_.get()) build_dn_->Deinit();

    if(set_dn_.get()) set_dn_->Deinit();
    if(get_dn_string_.get()) get_dn_string_->Deinit();
    if(get_dn_attribute_.get()) get_dn_attribute_->Deinit();
    if(get_dn_attribut_with_index_.get()) get_dn_attribut_with_index_->Deinit();
    if(set_dn_attribut_.get()) set_dn_attribut_->Deinit();

    if(get_path_limit_.get()) get_path_limit_->Deinit();
    if(is_ca_.get()) is_ca_->Deinit();
    if(subject_dn_.get()) subject_dn_->Deinit();
    if(authority_key_id_.get()) authority_key_id_->Deinit();
    if(start_time_.get()) start_time_->Deinit();
    if(end_time_.get()) end_time_->Deinit();
    if(get_status_.get()) get_status_->Deinit();
    if(issuer_dn_.get()) issuer_dn_->Deinit();
    if(serial_num_.get()) serial_num_->Deinit();
    if(subject_key_id_.get()) subject_key_id_->Deinit();
    if(verify_me_.get()) verify_me_->Deinit();
    if(x509_version_.get()) x509_version_->Deinit();
}

void X509CmClient::Stop() {
    stopped_ = true;
}

int32_t X509CmClient::ParseCert(const std::vector<uint8_t>& certMem, const uint32_t formatId,CryptoCertRef& cert_ref){
    CheckInit();
    auto parse_cert_request = std::make_shared<ParseCertRequest>();
    auto parse_cert_result = std::make_shared<ParseCertResult>();
    parse_cert_request->fire_forget() = false;
    parse_cert_request->certMem() = certMem;
    parse_cert_request->formatId() = formatId;
    int cm_res = parse_cert_->Request(parse_cert_request,parse_cert_result,500);
    CheckCommonResult(cm_res,"parse_cert_->Request");
    CRYP_INFO<<"parse_cert_result code:"<<parse_cert_result->code()<<" ref:"<<parse_cert_result->cert_ref().ref();
    if(0 == parse_cert_result->code()){
        cert_ref.ref = parse_cert_result->cert_ref().ref();
    }
    return parse_cert_result->code();
}

int32_t X509CmClient::CreateCSR(CipherCtxRef signerCtx, X509DNRef& derSubjectDN, std::map<std::uint32_t,
    std::string>& x509Extensions, std::uint32_t version, CmX509_Ref& csr){
    CheckInit();
    auto cert_sign_request = std::make_shared<CreateCertSignRequest>();
    auto cert_sign_result = std::make_shared<CreateCSRResult>();
    cert_sign_request->fire_forget() = false;
    cert_sign_request->version() = version;
    cert_sign_request->x509dn_ref().ref() = derSubjectDN.ref;
    cert_sign_request->x509Extensions() = x509Extensions;
    cert_sign_request->key().ref() = signerCtx.ref;
    int cm_res = create_cert_sign_request_->Request(cert_sign_request, cert_sign_result,500);
    CheckCommonResult(cm_res,"create_cert_sign_request_->Request");
    if(0 == cert_sign_result->code()){
        CRYP_INFO << "CreateCSR sucessed";
        csr = cert_sign_result->certSignRequest_ref();
    }
    return cert_sign_result->code();
}

int32_t X509CmClient::ExportCSR(CmX509_Ref&  csr_ref, std::vector<uint8_t>& csr_vec) {
    CheckInit();
    auto export_asn1_csr_request = std::make_shared<ExportASN1CertSignRequest>();
    auto export_asn1_csr_result = std::make_shared<ExportASN1CertSignResult>();
    export_asn1_csr_request->fire_forget() = false;
    export_asn1_csr_request->certSignRequest_ref().ref() = csr_ref.ref();
    int cm_res = export_asn1_cert_sign_request_->Request(export_asn1_csr_request,export_asn1_csr_result,500);
    CheckCommonResult(cm_res,"export_asn1_cert_sign_request_->Request");
    if(0 == export_asn1_csr_result->code()){
        CRYP_INFO << "ExportCSR sucessed";
        csr_vec = export_asn1_csr_result->signature();
    }
    return export_asn1_csr_result->code();
}

bool X509CmClient::ImportCert(CryptoCertRef& cert_ref, const std::string destCertPath){
    if(!inited_) return false;
    auto import_cert_request = std::make_shared<ImportCertRequest>();
    auto import_cert_result = std::make_shared<ImportCertResult>();
    import_cert_request->fire_forget() = false;
    import_cert_request->cert_ref().ref() = cert_ref.ref;
    import_cert_request->destCertPath() = destCertPath;
    auto cm_res = import_cert_->Request(import_cert_request,import_cert_result,500);
    if (cm_res != 0) {
        CRYP_INFO<<"import_cert_->Request failed.";
        return static_cast<int32_t>(CryptoErrc::kCommunicationError);
    }else{
        CRYP_INFO<<"import_cert_->Request success.";
    }
    return import_cert_result->result();
}

CertStatus X509CmClient::VerifyCert(CryptoCertRef& cert_ref,const std::string rootCertPath){
    if(!inited_) return CertStatus::kUnknown;
    auto verify_cert_request = std::make_shared<VerifyCertRequest>();
    auto verify_cert_result = std::make_shared<VerifyCertResult>();
    verify_cert_request->fire_forget() = false;
    verify_cert_request->cert_ref().ref() = cert_ref.ref;
    verify_cert_request->rootCertPath() = rootCertPath;
    auto cm_res = verify_cert_->Request(verify_cert_request,verify_cert_result,500);
    if (cm_res != 0) {
        CRYP_INFO << "verify_cert_->Request failed.";
        return CertStatus::kUnknown;
    } else {
        CRYP_INFO << "verify_cert_->Request success.";
    }
    return verify_cert_result->cert_status();
}


int32_t X509CmClient::FindCertByDn(const X509DN &subjectDn,const X509DN &issuerDn, time_t validityTimePoint,CryptoCertRef& cert_ref) noexcept{
    CheckInit();
    auto find_cert_bydn_request = std::make_shared<FindCertByDnRequest>();
    auto find_cert_bydn_result = std::make_shared<FindCertByDnResult>();
    find_cert_bydn_request->fire_forget() = false;
    find_cert_bydn_request->validityTimePoint() = validityTimePoint;

    for(auto it :subjectDn.attributeMap){
        std::pair<u_int32_t,std::string>temp;
        temp.first = static_cast<uint32_t>(it.first); 
        temp.second = it.second;
        find_cert_bydn_request->subjectDn().attributeMap().insert(temp);
    }

    for(auto it :issuerDn.attributeMap){
        std::pair<u_int32_t,std::string>temp;
        temp.first = static_cast<uint32_t>(it.first); 
        temp.second = it.second;
        find_cert_bydn_request->issuerDn().attributeMap().insert(temp);
    }
    auto cm_res = find_cert_by_dn_->Request(find_cert_bydn_request,find_cert_bydn_result,500);
    CheckCommonResult(cm_res,"find_cert_by_dn_->Request");
    return find_cert_bydn_result->code();

}

int32_t X509CmClient::BuildDn(std::string dn,X509DNRef& dn_ref) noexcept{
    CheckInit();
    auto build_dn_request = std::make_shared<BuildDnRequest>();
    auto build_dn_result = std::make_shared<BuildDnResult>();
    build_dn_request->fire_forget() = false;
    build_dn_request->dn() = dn;
    auto cm_res = build_dn_->Request(build_dn_request, build_dn_result,500);
    CheckCommonResult(cm_res,"build_dn_->Request");
    if(0 == build_dn_result->code()){
        CRYP_INFO << "BuildDn sucessed";
        dn_ref.ref = build_dn_result->x509dn_ref().ref();
    }
    return build_dn_result->code();
}

bool X509CmClient::SetDn(const std::string dn, X509DNRef& dn_ref){
    CRYP_INFO << "SetDn begin";
    if(!inited_) return false;
    auto set_dn_request = std::make_shared<SetDnRequest>();
    auto set_dn_result = std::make_shared<SetDnResult>();
    set_dn_request->fire_forget() = false;
    set_dn_request->dn() = dn;
    set_dn_request->x509dn_ref().ref() = dn_ref.ref;
    auto cm_res = set_dn_->Request(set_dn_request, set_dn_result,500);
    if(0 == cm_res){
        CRYP_INFO << "SetDn sucessed";
        return set_dn_result->Result();
    }else{
        return false;
    }
}

std::string X509CmClient::GetDnString(X509DNRef& dn_ref){
    std::string ret;
    if(!inited_) return ret;
    auto get_dn_string_request = std::make_shared<GetDnStringRequest>();
    auto get_dn_string_result = std::make_shared<DnCommonResult>();
    get_dn_string_request->fire_forget() = false;
    get_dn_string_request->x509dn_ref().ref() = dn_ref.ref;
    auto cm_res = get_dn_string_->Request(get_dn_string_request,get_dn_string_result,500);
    if(0 == cm_res){
        if(0 == get_dn_string_result->code()){
            ret = get_dn_string_result->dn_result();
        }
    }else{
       
    }
    return ret;
}

std::string X509CmClient::GetAttribute(const X509DN::AttributeId& id, X509DNRef& dn_ref){
    std::string ret;
    if(!inited_) return ret;
    auto get_dn_attribute_request = std::make_shared<GetAttributeRequest>();
    auto get_dn_attribute_result = std::make_shared<DnCommonResult>();
    get_dn_attribute_request->fire_forget() = false;
    get_dn_attribute_request->id() = static_cast<AttributeId>(id);
    get_dn_attribute_request->x509dn_ref().ref() = dn_ref.ref;
    auto cm_res = get_dn_attribute_->Request(get_dn_attribute_request,get_dn_attribute_result,500);
    if(0 == cm_res){
        if(0 == get_dn_attribute_result->code()){
            ret = get_dn_attribute_result->dn_result();
        }
    }else{
       
    }
    return ret;
}

std::string X509CmClient::GetAttributeWithIndex(const X509DN::AttributeId& id, const uint32_t& index,const X509DNRef& dn_ref){
    std::string ret;
    if(!inited_) return ret;
    auto get_dn_attribute_with_index_request = std::make_shared<GetAttributeWithIndexRequest>();
    auto get_dn_attribute_with_index_result = std::make_shared<DnCommonResult>();
    get_dn_attribute_with_index_request->fire_forget() = false;
    get_dn_attribute_with_index_request->id() = static_cast<AttributeId>(id);
    get_dn_attribute_with_index_request->index() = index;
    get_dn_attribute_with_index_request->x509dn_ref().ref() = dn_ref.ref;
    auto cm_res = get_dn_attribut_with_index_->Request(get_dn_attribute_with_index_request,get_dn_attribute_with_index_result,500);
    if(0 == cm_res){
        if(0 == get_dn_attribute_with_index_result->code()){
            ret = get_dn_attribute_with_index_result->dn_result();
        }
    }else{
       
    }
    return ret;
}

bool X509CmClient::SetAttribute(const X509DN::AttributeId& id,const std::string& attribute, X509DNRef& dn_ref){
    if(!inited_) return false;
    auto set_dn_attribute_request = std::make_shared<SetAttributeRequest>();
    auto set_dn_attribute_result = std::make_shared<SetAttributeResult>();
    set_dn_attribute_request->fire_forget() = false;
    set_dn_attribute_request->id() = static_cast<AttributeId>(id);
    set_dn_attribute_request->x509dn_ref().ref() = dn_ref.ref;
    auto cm_res = set_dn_attribut_->Request(set_dn_attribute_request,set_dn_attribute_result,500);
    if(0 == cm_res){
        return set_dn_attribute_result->result();
    }else{
        return false;
    }
}


uint32_t X509CmClient::GetPathLimit(CryptoCertRef& cert_ref){
    uint32_t ret = 0;
    if (!inited_) return ret;
    auto request = std::make_shared<GetPathLimitRequest>();
    auto result = std::make_shared<GetPathLimitResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = get_path_limit_->Request(request, result, 500);
    if (0 == cm_res) {
        ret = result->count();
    } else {

    }
    return ret;
}

bool X509CmClient::IsCa(CryptoCertRef& cert_ref){
    bool ret = false;
    if (!inited_) return ret;
    auto is_ca_request = std::make_shared<IsCaRequest>();
    auto is_ca_result = std::make_shared<CertCommonResult>();
    is_ca_request->fire_forget() = false;
    is_ca_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = is_ca_->Request(is_ca_request, is_ca_result, 500);
    if (0 == cm_res) {
        ret = is_ca_result->result();
    } else {
    }
    return ret;
}

X509DNRef X509CmClient::SubjectDn(CryptoCertRef& cert_ref){
    X509DNRef ret;
    if (!inited_) return ret;
    auto request = std::make_shared<SubjectDnRequest>();
    auto result = std::make_shared<SubjectDnResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = subject_dn_->Request(request, result, 500);
    if (0 == cm_res) {
        ret.ref = result->x509_dn_ref().ref();
    } else {
    }
    return ret;
}

std::string X509CmClient::AuthorityKeyId(CryptoCertRef& cert_ref){
    std::string ret;
    if(!inited_) return ret;
    auto auth_key_request = std::make_shared<AuthorityKeyIdRequest>();
    auto auth_key_result = std::make_shared<AuthorityKeyIdResult>();
    auth_key_request->fire_forget() = false;
    auth_key_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = authority_key_id_->Request(auth_key_request,auth_key_result,500);
    if(0 == cm_res){
        ret = auth_key_result->key_id();
    }else{

    }
    return ret;
}


time_t X509CmClient::StartTime(CryptoCertRef& cert_ref){
    std::int64_t ret = 0;
    if(!inited_) return static_cast<time_t>(ret);
    auto start_time_request = std::make_shared<StartTimeRequest>();
    auto start_time_result = std::make_shared<StartTimeResult>();
    start_time_request->fire_forget() = false;
    start_time_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = start_time_->Request(start_time_request,start_time_result,500);
    if(0 == cm_res){
        ret = start_time_result->time();
    }else{

    }
    return static_cast<time_t>(ret);
}

time_t X509CmClient::EndTime(CryptoCertRef& cert_ref){
    std::int64_t ret = 0;
    if(!inited_) return static_cast<time_t>(ret);
    auto end_time_request = std::make_shared<EndTimeRequest>();
    auto end_time_result = std::make_shared<EndTimeResult>();
    end_time_request->fire_forget() = false;
    end_time_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = end_time_->Request(end_time_request,end_time_result,500);
    if(0 == cm_res){
        ret = end_time_result->time();
    }else{

    }
    return static_cast<time_t>(ret);
}

Certificate::Status X509CmClient::GetStatus(CryptoCertRef& cert_ref){
    std::uint32_t ret = 2;
    if(!inited_) return static_cast<Certificate::Status>(ret);
    auto get_status_request = std::make_shared<GetStatusRequest>();
    auto get_status_result = std::make_shared<GetStatusResult>();
    get_status_request->fire_forget() = false;
    get_status_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = get_status_->Request(get_status_request,get_status_result,500);
    if(0 == cm_res){
        ret = get_status_result->status();
    }else{
    
    }
    return static_cast<Certificate::Status>(ret);
}

bool X509CmClient::IsRoot(CryptoCertRef& cert_ref){
    bool ret = false;
    if(!inited_) return ret;
    auto is_root_request = std::make_shared<IsRootRequest>();
    auto is_root_result = std::make_shared<CertCommonResult>();
    is_root_request->fire_forget() = false;
    is_root_request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = is_root_->Request(is_root_request,is_root_result,500);
    if(0 == cm_res){
        ret = is_root_result->result();
    }else{

    }
    return ret;
}

X509DNRef X509CmClient::IssuerDn(CryptoCertRef& cert_ref){
    X509DNRef ret;
    if (!inited_) return ret;
    auto request = std::make_shared<IssuerDnRequest>();
    auto result = std::make_shared<IssuerDnResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = issuer_dn_->Request(request, result, 500);
    if (0 == cm_res) {
        ret.ref = result->issuer_dn_ref().ref();
    } else {
    }
    return ret;
}

std::string X509CmClient::SerialNumber(CryptoCertRef& cert_ref){
    std::string ret;
    if (!inited_) return ret;
    auto request = std::make_shared<SerialNumberRequest>();
    auto result = std::make_shared<SerialNumberResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = serial_num_->Request(request, result, 500);
    if (0 == cm_res) {
        ret = result->x509_dn();
    } else {
    }
    return ret;
}

std::string X509CmClient::SubjectKeyId(CryptoCertRef& cert_ref){
    std::string ret;
    if (!inited_) return ret;
    auto request = std::make_shared<SubjectKeyIdRequest>();
    auto result = std::make_shared<SubjectKeyIdResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = subject_key_id_->Request(request, result, 500);
    if (0 == cm_res) {
        ret = result->subjectkey_id();
    } else {
    }
    return ret;
}
bool X509CmClient::VerifyMe(CryptoCertRef& caCert){
    bool ret = false;
    if (!inited_) return ret;
    auto request = std::make_shared<VerifyMeRequest>();
    auto result = std::make_shared<CertCommonResult>();
    request->fire_forget() = false;

    request->cert_ref().ref() = caCert.ref;
    auto cm_res = verify_me_->Request(request, result, 500);
    if (0 == cm_res) {
        ret = result->result();
    } else {

    }
    return ret;
}
std::uint32_t X509CmClient::X509Version(CryptoCertRef& cert_ref){
    std::uint32_t ret = 0;
    if (!inited_) return ret;
    auto request = std::make_shared<X509VersionRequest>();
    auto result = std::make_shared<X509VersionResult>();
    request->fire_forget() = false;
    request->cert_ref().ref() = cert_ref.ref;
    auto cm_res = x509_version_->Request(request, result, 500);
    if (0 == cm_res) {
        ret = result->version();
    } else {

    }
    return ret;
}

// int32_t X509CmClient::ReleaseObject(uint64_t ref) {
//     if (!inited_) {
//         return static_cast<int32_t>(CryptoErrc::kInvalidUsageOrder);
//     }

//     auto release_object_request = std::make_shared<ReleaseObjectRequest>();
//     release_object_request->ref() = ref;
//     release_object_request->fire_forget(false);
//     auto release_object_result = std::make_shared<ReleaseObjectResult>();
//     int cm_res = release_object_method_->Request(release_object_request, release_object_result, 500);

//     if (cm_res != 0) {
//         return static_cast<int32_t>(CryptoErrc::kCommunicationError);
//     }

//     return static_cast<int32_t>(release_object_result->code());
// }

X509CmClient::X509CmClient() {
    Init();
}

X509CmClient::~X509CmClient() {
    Stop();
    Deinit();
}

}
}
}
}