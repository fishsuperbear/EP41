#include <dlfcn.h>
#include "dl_adf/include/library_loader.h"
#include "dl_adf/include/log.h"

namespace hozon {
namespace netaos {
namespace dl_adf {
int32_t LibraryLoader::Load(const std::string& lib_path) { 
    if(_handle_map.find(lib_path) == _handle_map.end()) {
        DL_EARLY_LOG << "Start dlopen path: " << lib_path;
        void* handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if(handle == nullptr) {
            const char* err = dlerror();
            DL_EARLY_LOG << "Fail to open: " << lib_path << " err: " << err;
            return -1;
        }
        _handle_map[lib_path] = handle;
        DL_EARLY_LOG << "handle: " << handle;
    }
    return 0;
}

int32_t LibraryLoader::Unload(const std::string& lib_path) {
    if(_handle_map.find(lib_path) == _handle_map.end()) {
        DL_EARLY_LOG << "Fail to unload: " << lib_path;
        return -1;
    }
    else if(nullptr != _handle_map[lib_path]) {
        DL_EARLY_LOG << "begin close lib: " << lib_path << " handle: " << _handle_map[lib_path];
        dlclose(_handle_map[lib_path]);
        DL_EARLY_LOG << "close lib: " << lib_path << " handle: " << _handle_map[lib_path];
        // _handle_map.erase(lib_path);
    }
    else {

    }
    return 0;
}

int32_t LibraryLoader::UnloadAll() {
    for(auto handle_map : _handle_map) {
        if(nullptr != handle_map.second) {
            dlclose(handle_map.second);
        }
    }
    _handle_map.clear();
    DL_EARLY_LOG << "handle_map size :" <<  _handle_map.size();
    return 0;
}

void* LibraryLoader::GetSymbol(const std::string& symbol_name, 
                                const std::string& lib_path) {
    if(_handle_map.find(lib_path) == _handle_map.end()) {
        DL_EARLY_LOG << "Fail to find library: " << lib_path;  
        return nullptr;
    }                  
    else if(nullptr != _handle_map[lib_path]) {
        void* sym = dlsym(_handle_map[lib_path], symbol_name.c_str());
        return sym;
    } 
    else {

    }   
    DL_EARLY_LOG << "Fail to find symbol: " << symbol_name << " in path: " \
        << lib_path;   
    return nullptr;       
}

}   // namespace dl_adf
}   // namespace netaos
}   // namespace hozon