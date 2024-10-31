#pragma once
#include <string>
#include <iostream>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace dl_adf {
class LibraryLoader {
public:
    LibraryLoader()  = default;
    ~LibraryLoader()  = default;
    int32_t Load(const std::string& lib_path);
    int32_t Unload(const std::string& lib_path);
    int32_t UnloadAll();
    void* GetSymbol(const std::string& symbol_name,
                     const std::string& lib_path);
private:
    // void* _handle = nullptr;
    std::unordered_map<std::string, void*> _handle_map;
};
}   // namespace dl_adf
}   // namespace netaos
}   // namespace hozon