#pragma once

#include "adf-lite/include/base.h"
// cuda includes
#include "cuda_runtime_api.h"
#include "cuda.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
using ReleaseCB = std::function<void(bool,void*)>;

struct NvsImageCUDA : public BaseData {
public:
    NvsImageCUDA() {}
    NvsImageCUDA(const NvsImageCUDA& another) = delete;
    NvsImageCUDA& operator=(const NvsImageCUDA& another) = delete;
    void SetReleaseCB(ReleaseCB i_prelease_cb) { release_manager = i_prelease_cb; }

    ~NvsImageCUDA() {
        if (cuda_dev_ptr != nullptr) {
            if (release_manager != nullptr) {
                release_manager(need_user_free,cuda_dev_ptr);
                return;
            }
            if(need_user_free){
                cudaFree(cuda_dev_ptr);
            }
        }
    }

    double data_time_sec;
    double virt_time_sec;
    uint32_t width;
    uint32_t height;
    std::string format;
    uint64_t size;
    uint64_t step;
    void* cuda_dev_ptr;
    bool need_user_free;
private:
    ReleaseCB release_manager = nullptr;
};

}
}
}