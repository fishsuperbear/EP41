#ifndef HW_NVMEDIA_MULTIIPC_CONSUMER_ENC_H
#define HW_NVMEDIA_MULTIIPC_CONSUMER_ENC_H

#include "hw_nvmedia_common.h"

__BEGIN_DECLS

#ifndef HW_MODULE_API_VERSION
#define HW_MODULE_API_VERSION HW_MAKEV_MODULE_API_VERSION(0, 1)
#endif

typedef struct hw_nvmedia_multiipc_consumer_enc_module_t {
    struct hw_module_t common;
    struct hw_nvmedia_module_t nvmedia;
} hw_nvmedia_multiipc_consumer_enc_module_t;

__END_DECLS

#endif  // HW_NVMEDIA_MULTIIPC_CONSUMER_ENC_H
