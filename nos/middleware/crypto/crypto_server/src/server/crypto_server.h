#pragma once

#include <atomic>
#include <memory>
#include "common/entry_point.h"

namespace hozon {
namespace netaos {
namespace crypto {

class CryptoServer {

public:
    static CryptoServer& Instance();
    static void Destroy();

    bool Init();
    void Deinit();
    void Start();
    void Stop();
    std::unique_ptr<hozon::netaos::crypto::cryp::CryptoProvider> crypto_provider_;
    std::unique_ptr<hozon::netaos::crypto::keys::KeyStorageProvider> key_storage_provider_;
    // std::unique_ptr<hozon::netaos::crypto::x509::X509Provider> x509_provider_;

private:
    CryptoServer();
    ~CryptoServer();

    std::atomic<bool> stopped_{false};
    std::once_flag onceFlag;

};

}
}
}