
#ifndef PHM_FAULT_MANAGER_H
#define PHM_FAULT_MANAGER_H

#include <mutex>
#include <memory>

namespace hozon {
namespace netaos {
namespace phm_server {

class PhmCollectFile;
class FaultManager {

public:
    static FaultManager* getInstance();

    void Init();
    void DeInit();

private:
    FaultManager();
    FaultManager(const FaultManager &);
    FaultManager & operator = (const FaultManager &);

private:
    static std::mutex mtx_;
    static FaultManager* instance_;
    std::shared_ptr<PhmCollectFile> m_spPhmCollectFile;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_MANAGER_H