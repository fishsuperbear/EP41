#pragma once
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <syscall.h>

namespace hozon {
namespace netaos {
namespace dl_adf {

class DlLogger {
public: 
    DlLogger(){};
    ~DlLogger() {
        std::cout << std::endl;
    }

    template <typename T>
    DlLogger & operator<<(const T& value) {
        std::cout << value;
        return *this;
    }  
};

#define LOG_HEAD    getpid() << " " << (long int)syscall(__NR_gettid) \
         << " " << __FUNCTION__ << "@" << strrchr(__FILE__, '/') + 1 << "(" << __LINE__ << ") | "

#define DL_EARLY_LOG            DlLogger() << LOG_HEAD
}   // namespace dl_adf
}   // namespace netaos
}   // namespace hozon


