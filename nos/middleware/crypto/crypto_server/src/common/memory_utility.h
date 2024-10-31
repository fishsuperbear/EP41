#pragma once

#include <memory>

namespace hozon {
namespace netaos {
namespace crypto {

struct OsslMemDeleter {
    void operator() (unsigned char *p) const {
        if (p) {
            OPENSSL_free(p);
            p = nullptr;
        }
    }
};

using OsslMemSptr = std::shared_ptr<unsigned char>;
#define MakeSharedOsslMem(size) hozon::netaos::crypto::OsslMemSptr(static_cast<unsigned char*>(::OPENSSL_malloc(size)), hozon::netaos::crypto::OsslMemDeleter())

}
}
}