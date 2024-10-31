#ifndef HALNODE_IMPL_H
#define HALNODE_IMPL_H

#include "halnode.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h> /* For size_t */
#include <stdbool.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <signal.h>
#include <linux/ioctl.h>
#include <linux/types.h>

#define NETA_CDEV_PATH "/dev/netahal"

// #pragma pack(8)

typedef struct  {

        consumer_info consumerinfo;
        sensor_info sensorinfo;
        u32 sensor_id;
        u32 flags;  //0 producer 1 consumer
        u32 active;  //0 noactive  1 active
        u32 chanel_id; //0-5
}mem_mapinfo;

// #pragma pack()

/* Driver IOCTL codes*/
#define NETA_IO_DRV_MAGIC 'm'

#define NETA_IO_PRODUCER \
    _IOWR(NETA_IO_DRV_MAGIC, 1, mem_mapinfo)

#define NETA_IO_CONSUMER \
    _IOWR(NETA_IO_DRV_MAGIC, 2, mem_mapinfo)

#define NETA_IO_SET_PRODUCER_INFO \
    _IOWR(NETA_IO_DRV_MAGIC, 3, mem_mapinfo)

#define NETA_IO_SET_CONSUMER_INFO \
    _IOWR(NETA_IO_DRV_MAGIC, 4, mem_mapinfo)

#endif
