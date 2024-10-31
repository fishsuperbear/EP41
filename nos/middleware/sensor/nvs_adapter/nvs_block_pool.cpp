#include "sensor/nvs_adapter/nvs_block_pool.h"

namespace hozon {
namespace netaos {
namespace nv { 

int32_t NVSBlockPool::Create(uint32_t num_packet) {
    name = "Pool";
    _num_packet = num_packet;

    NvSciError err = NvSciStreamStaticPoolCreate(num_packet, &block);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create pool block, ret " << LogHexNvErr(err); 
        return -1;
    }

    RegIntoEventService();
    return 0;
}

int32_t NVSBlockPool::OnConnected() {
    NvSciError err = NvSciStreamBlockConsumerCountGet(block, &_num_consumers);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create query number of consumers, ret " << LogHexNvErr(err); 
        return -1;
    }

    NVS_LOG_INFO << "Pool connected, consumer num " << _num_consumers;

    return 0;
}

int32_t NVSBlockPool::OnElements() {
    NvSciError err;

    /* Query producer element count */
    err = NvSciStreamBlockElementCountGet(block,
                                          NvSciStreamBlockType_Producer,
                                          &_num_producer_elem);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to query producer element count, ret " << err;
        return -1;
    }
    NVS_LOG_INFO << "Producer element count " << _num_producer_elem;

    /* Query consumer element count */
    err = NvSciStreamBlockElementCountGet(block,
                                          NvSciStreamBlockType_Consumer,
                                          &_num_consumer_elem);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to query consumer element count, ret " << err;
        return -1;
    }
    NVS_LOG_INFO << "Consumer element count " << _num_consumer_elem;

    /* Query all producer elements */
    for (uint32_t i=0U; i<_num_producer_elem; ++i) {
        err = NvSciStreamBlockElementAttrGet(block,
                                             NvSciStreamBlockType_Producer, i,
                                             &_producer_elem[i].userName,
                                             &_producer_elem[i].attrList);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to query producer element, ret " << err;
            return -1;
        }
    }

    /* Query all consumer elements */
    for (uint32_t i=0U; i<_num_consumer_elem; ++i) {
        err = NvSciStreamBlockElementAttrGet(block,
                                             NvSciStreamBlockType_Consumer, i,
                                             &_consumer_elem[i].userName,
                                             &_consumer_elem[i].attrList);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to query consumer element, ret " << err;
            return -1;
        }
    }

    /* Indicate that all element information has been imported */
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to complete element import, ret " << err;
        return -1;
    }

    /*
     * Go through requested elements from producer and consumer and line
     *   them up. A general streaming application might not have a one to
     *   one correspondence, and the pool may have to decide what subset
     *   of elements to select based on knowledge of the data types that
     *   the application suite supports. This sample application is much
     *   simpler, but we still go through the process rather than assuming
     *   producer and consumer have requested the same things in the same
     *   order.
     */
    uint32_t numElem = 0, p, c, e, i;
    ElemAttr elem[MAX_ELEMS];
    for (p=0; p<_num_producer_elem; ++p) {
        ElemAttr* prodElem = &_producer_elem[p];
        for (c=0; c<_num_consumer_elem; ++c) {
            ElemAttr* consElem = &_consumer_elem[c];

            /* If requested element types match, combine the entries */
            if (prodElem->userName == consElem->userName) {
                ElemAttr* poolElem = &elem[numElem++];
                poolElem->userName = prodElem->userName;
                poolElem->attrList = NULL;

                // if (ELEMENT_NAME_IMAGE == prodElem->userName) {
                //     PrintBufAttrs(prodElem->attrList);
                //     PrintBufAttrs(consElem->attrList);
                // }

                /* Combine and reconcile the attribute lists */
                NvSciBufAttrList oldAttrList[2] = { prodElem->attrList,
                                                    consElem->attrList };
                NvSciBufAttrList conflicts = NULL;
                err = NvSciBufAttrListReconcile(oldAttrList, 2,
                                                &poolElem->attrList,
                                                &conflicts);

                // if ((ELEMENT_NAME_IMAGE == prodElem->userName) && conflicts) {
                //     NvsUtility::PrintBufAttrs(conflicts);
                // }

                /* Discard any conflict list.
                 *  (Could report its contents for additional debug info)
                 */
                if (NULL != conflicts) {
                    NvSciBufAttrListFree(conflicts);
                }

                /* Abort on error */
                if (NvSciError_Success != err) {
                    NVS_LOG_CRITICAL << "Fail to reconcile element " << log::loghex(poolElem->userName) << ", ret " << err;
                    return -1;
                }

                /* Found a match for this producer element so move on */
                break;
            }  /* if match */
        } /* for all requested consumer elements */
    } /* for all requested producer elements */

    /* Should be at least one element */
    if (0 == numElem) {
        NVS_LOG_CRITICAL << "Fail to find any common elements";
        return -1;
    }

    /* The requested attribute lists are no longer needed, so discard them */
    for (p=0; p<_num_producer_elem; ++p) {
        ElemAttr* prodElem = &_producer_elem[p];
        if (NULL != prodElem->attrList) {
            NvSciBufAttrListFree(prodElem->attrList);
            prodElem->attrList = NULL;
        }
    }
    for (c=0; c<_num_consumer_elem; ++c) {
        ElemAttr* consElem = &_consumer_elem[c];
        if (NULL != consElem->attrList) {
            NvSciBufAttrListFree(consElem->attrList);
            consElem->attrList = NULL;
        }
    }

    /* Inform the stream of the chosen elements */
    for (e=0; e<numElem; ++e) {
        ElemAttr* poolElem = &elem[e];
        err = NvSciStreamBlockElementAttrSet(block,
                                             poolElem->userName,
                                             poolElem->attrList);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to send element " << log::loghex(poolElem->userName) << " info, ret " << LogHexNvErr(err);
            return -1;
        }
    }

    /* Indicate that all element information has been exported */
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_ElementExport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to complete element export, ret " << LogHexNvErr(err);
        return -1;
    }

    /*
     * Create and send all the packets and their buffers
     * Note: Packets and buffers are not guaranteed to be received by
     *       producer and consumer in the same order sent, nor are the
     *       status messages sent back guaranteed to preserve ordering.
     *       This is one reason why an event driven model is more robust.
     */
    for (i=0; i<_num_packet; ++i) {

        /*
         * Create a new packet
         * Our pool implementation doesn't need to save any packet-specific
         *   data, but we do need to provide unique cookies, so we just
         *   use the pointer to the location we save the handle. For other
         *   blocks, this will be a pointer to the structure where the
         *   packet information is kept.
         */
        NvSciStreamCookie cookie = (NvSciStreamCookie)&_packet[i];
        err = NvSciStreamPoolPacketCreate(block,
                                          cookie,
                                          &_packet[i]);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to complete element export, ret " << LogHexNvErr(err);
            return 0;
        }

        /* Create buffers for the packet */
        for (e=0; e<numElem; ++e) {
            /* Allocate a buffer object */
            NvSciBufObj obj;
            err = NvSciBufObjAlloc(elem[e].attrList, &obj);
            if (NvSciError_Success != err) {
                NVS_LOG_CRITICAL << "Fail to allocate buffer " << i << " of element " << e << ", ret " << LogHexNvErr(err);
                return -1;
            }

            /* Insert the buffer in the packet */
            err = NvSciStreamPoolPacketInsertBuffer(block,
                                                    _packet[i],
                                                    e, obj);
            if (NvSciError_Success != err) {
                NVS_LOG_CRITICAL << "Fail to insert buffer " << i << " of element " << e << ", ret " << LogHexNvErr(err);
                return -1;
            }

            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
        }

        /* Indicate packet setup is complete */
        err = NvSciStreamPoolPacketComplete(block,
                                            _packet[i]);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to complete packet " << i << ", ret " << LogHexNvErr(err);
            return -1;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the status has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_PacketExport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to complete packet export, ret " << LogHexNvErr(err);
        return -1;
    }

    /* Once all packets are set up, no longer need to keep the attributes */
    for (e=0; e<numElem; ++e) {
        ElemAttr* poolElem = &elem[e];
        if (NULL != poolElem->attrList) {
            NvSciBufAttrListFree(poolElem->attrList);
            poolElem->attrList = NULL;
        }
    }

    return 0;
}

int32_t NVSBlockPool::OnPacketStatus() {
    if (++_num_packet_ready < _num_packet) {
        return 0;
    }

    bool packetFailure = false;
    NvSciError err;

    /* Check each packet */
    for (uint32_t p = 0; p < _num_packet; ++p) {
        /* Check packet acceptance */
        bool accept;
        err = NvSciStreamPoolPacketStatusAcceptGet(block,
                                                   _packet[p],
                                                   &accept);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to retrieve packet " << p << " acceptance-statue, ret " << LogHexNvErr(err);
            return -1;
        }

        if (accept) {
            continue;
        }

        /* On rejection, query and report details */
        packetFailure = true;
        NvSciError status;

        /* Check packet status from producer */
        err = NvSciStreamPoolPacketStatusValueGet(
                block,
                _packet[p],
                NvSciStreamBlockType_Producer, 0U,
                &status);
        if (NvSciError_Success != err) {
            NVS_LOG_CRITICAL << "Fail to retrieve packet " << p << " statue from producer, ret " << LogHexNvErr(err);
            return -1;
        }
        if (status != NvSciError_Success) {
            NVS_LOG_CRITICAL << "Producer rejected packet " << p << " with error " << LogHexNvErr(status);
        }

        /* Check packet status from consumers */
        for (uint32_t c = 0; c < _num_consumers; ++c) {
            err = NvSciStreamPoolPacketStatusValueGet(
                    block,
                    _packet[p],
                    NvSciStreamBlockType_Consumer, c,
                    &status);
            if (NvSciError_Success != err) {
                NVS_LOG_CRITICAL << "Fail to retrieve packet " << p << " statue from consumer " << c << ", ret " << LogHexNvErr(err);
                return -1;
            }
            if (status != NvSciError_Success) {
                NVS_LOG_CRITICAL << "Consumer " << c << " rejected packet " << p << " with error " << LogHexNvErr(status);
            }
        }
    }

    /* Indicate that status for all packets has been received. */
    // poolData->packetsDone = true;
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_PacketImport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to complete packets export " << LogHexNvErr(err);
        return 0;
    }

    return packetFailure ? -1 : 0;
}

}
}
}