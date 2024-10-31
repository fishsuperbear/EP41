#include "adf-lite/include/classloader.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

namespace hozon {
namespace netaos {
namespace adf_lite {

int32_t LibraryLoader::Load(const std::string& lib_path) {
    _lib_path = lib_path;
    _handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    ADF_INTERNAL_LOG_DEBUG << "LibraryLoader lib_path is " << lib_path;
    if (_handle == nullptr) {
        ADF_INTERNAL_LOG_ERROR << "dlerror " << dlerror();
        return -1;
    }

    return 0;
}

void LibraryLoader::Unload() {
    if (_handle) {
        int res = dlclose(_handle);
        ADF_INTERNAL_LOG_DEBUG << "LibraryLoader lib_path is " << _lib_path << " Unload, res = " << res;
        _handle = nullptr;
    }
}

void* LibraryLoader::GetSymbol(const std::string& symbol_name) {
    void* sym = dlsym(_handle, symbol_name.c_str());
    return sym;
}

}
}
}