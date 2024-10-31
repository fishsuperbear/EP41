#ifndef __LOGLOCATION_H__
#define __LOGLOCATION_H__

#include <cstring>
#include <iostream>

namespace hozon {
namespace netaos {
namespace log {

class SourceLocation {
   public:
    SourceLocation(const char* file, int line) {
        auto len = std::strlen(file);
        if (len > kMaxFileNameSize)
            fileName = file + (len - kMaxFileNameSize);
        else
            fileName = file;
        lineNo = line;
    }

    ~SourceLocation() {}

    const char* fileName;
    int lineNo;

   private:
    static constexpr size_t kMaxFileNameSize{18};
};

#define FROM_HERE SourceLocation(__FILE__, __LINE__)

}  // namespace log
}  // namespace netaos
}  // namespace hozon
#endif  // __LOGLOCATION_H__
