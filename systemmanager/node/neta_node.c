#include "neta_node.h"

struct neta_node        __neta_node = 
{
    .lognode = {
        .pprocentry_logbase = NULL,
        .pprocentry_logglobalinfo = NULL,
        .plogblockdevnode = NULL,
    },
    .halnode = {
        .pprocentry_halbase = NULL,
        .pprocentry_halglobalinfo = NULL,
    },
};


