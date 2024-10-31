#pragma once

#include <stdint.h>
#include <mutex>
#include <string>
#include <vector>

namespace hozon {
namespace netaos {
namespace update {

class StateFileManager {
public:

    static StateFileManager* Instance();
    void Init();
    void Deinit();
    bool CreateStateFile();
    bool UpdateStateFile(const std::string& stateContent);
    bool ReadStateFile(std::string& stateContent);

private:
    StateFileManager();
    ~StateFileManager();
    StateFileManager(const StateFileManager &);
    StateFileManager & operator = (const StateFileManager &);

    static std::mutex m_mtx;
    static StateFileManager* m_pInstance;

};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
