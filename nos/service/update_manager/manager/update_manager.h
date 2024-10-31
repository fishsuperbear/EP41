
#ifndef UPDATE_MANAGER_H
#define UPDATE_MANAGER_H

#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

class UpdateManager {

public:
    UpdateManager();
    ~UpdateManager();
    void Init();
    void Start();
    void Run();
    void Stop();
    void Deinit();
    
private:
    UpdateManager(const UpdateManager &);
    UpdateManager & operator = (const UpdateManager &);

    bool stop_flag_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_MANAGER_H
