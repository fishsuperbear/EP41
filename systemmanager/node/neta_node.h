#ifndef NETA_NODE_H
#define NETA_NODE_H

#include "neta_node_common.h"
#include "neta_log.h"
#include "neta_halnode.h"

typedef struct neta_node
{
    struct neta_lognode        lognode;
    struct neta_halnode        halnode;
} neta_node;

extern struct neta_node        __neta_node;

#endif
