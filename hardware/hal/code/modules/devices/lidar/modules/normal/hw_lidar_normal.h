#ifndef HW_LIDAR_SOCKET_ROBOSENSEM1_H
#define HW_LIDAR_SOCKET_ROBOSENSEM1_H

#include "lidar/modules/normal/impl/hw_lidar_normal_impl.h"

__BEGIN_DECLS

#define HW_LIDAR_MODULE_API_VERSION HW_MAKEV_MODULE_API_VERSION(0, 1)

typedef struct hw_lidar_module_t
{
	struct hw_module_t common;
} hw_lidar_module_t;

__END_DECLS

#endif // HW_LIDAR_SOCKET_ROBOSENSEM1_H