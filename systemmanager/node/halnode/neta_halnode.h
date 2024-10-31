#ifndef NETA_HALNODE_H
#define NETA_HALNODE_H

#include "neta_klog.h"

#define NETA_HALBASENODE_NAME              "netahal"
#define NETA_HALGLOBALINFONODE_NAME        "global_info"

typedef struct neta_halnode
{
    struct proc_dir_entry*      pprocentry_halbase;
    // output dynamic info about hal
    struct proc_dir_entry*      pprocentry_halglobalinfo;
} neta_halnode;

#endif
