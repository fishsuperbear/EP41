#pragma once

#include "nvscibuf.h"

namespace hozon {
namespace netaos {
namespace nv {

class NvsUtility {
public:
    static void PrintBufAttrs(NvSciBufAttrList buf_attrs);
    static const char* GetKeyName(int key);
};

}
}
}