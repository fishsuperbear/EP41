#ifndef HW_NVMEDIA_GROUPA_H
#define HW_NVMEDIA_GROUPA_H

#include "hw_nvmedia_common.h"

__BEGIN_DECLS

#define HW_MODULE_API_VERSION			HW_MAKEV_MODULE_API_VERSION(0,1)

typedef struct hw_nvmedia_groupa_module_t {
	struct hw_module_t				common;
	struct hw_nvmedia_module_t		nvmedia;
} hw_nvmedia_groupa_module_t;

__END_DECLS

#endif
