#include "crypto_server.h"
#include <mutex>
#include <memory>
#include "crypto_cm_server.h"
#include "crypto_server.h"
#include "x509_cm_server.h"

namespace hozon {
namespace netaos {
namespace crypto {

static CryptoServer* sinstance_ = nullptr;
static std::mutex sinstance_mutex_;

CryptoServer& CryptoServer::Instance()  {

    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (!sinstance_) {
        sinstance_ = new CryptoServer();
    }

    return *sinstance_;
}

void CryptoServer::Destroy()  {

    std::lock_guard<std::mutex> lock(sinstance_mutex_);
    if (sinstance_) {
        delete sinstance_;
    }
}

bool CryptoServer::Init()  {
    std::call_once(onceFlag, [&](){
        // Load provider and keep only one instance in life cycle of the process.
        crypto_provider_ = hozon::netaos::crypto::LoadCryptoProvider();
        key_storage_provider_ = hozon::netaos::crypto::LoadKeyStorageProvider();
        // x509_provider_ = hozon::netaos::crypto::LoadX509Provider();
        // hozon::netaos::crypto::CryptoCmServer::Instance().SetCryptoProvider(crypto_provider_.get());
    });
    return false;
}

void CryptoServer::Deinit()  {
    if (crypto_provider_) {
        crypto_provider_.reset(nullptr);
    }
    if (key_storage_provider_) {
         key_storage_provider_.reset(nullptr);
    }
    // if (x509_provider_) {
    //     x509_provider_.reset(nullptr);
    // }
}

void CryptoServer::Start() {
    CryptoCmServer::Instance().Start();
    X509CmServer::Instance().Start();
}

void CryptoServer::Stop() {
    stopped_ = true;
    CryptoCmServer::Instance().Stop();
    X509CmServer::Instance().Stop();
}

CryptoServer::CryptoServer() {

}

CryptoServer::~CryptoServer() {

}

}
}
}