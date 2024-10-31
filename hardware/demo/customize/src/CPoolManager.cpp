// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software, related documentation and any
// modifications thereto. Any use, reproduction, disclosure or distribution
// of this software and related documentation without an express license
// agreement from NVIDIA Corporation is strictly prohibited.

#include "CPoolManager.hpp"

CPoolManager::CPoolManager(NvSciStreamBlock handle, uint32_t uSensor, uint32_t numPackets)
    : CEventHandler("Pool", handle, uSensor)
{
    m_handle = handle;
    m_numPacketReady = 0;
    m_numPackets = numPackets;
    m_elementsDone = false;
    m_packetsDone = false;
}

CPoolManager::~CPoolManager(void)
{
    LOG_DBG("Pool release.\n");

    for (uint32_t i = 0U; i < m_numProdElem; i++) {
        if (m_prodElems[i].bufAttrList != nullptr) {
            NvSciBufAttrListFree(m_prodElems[i].bufAttrList);
        }
    }
    for (uint32_t i = 0U; i < m_numConsElem; i++) {
        if (m_consElems[i].bufAttrList != nullptr) {
            NvSciBufAttrListFree(m_consElems[i].bufAttrList);
        }
    }

    for (uint32_t i = 0U; i < m_numElem; i++) {
        if (m_elems[i].bufAttrList != nullptr) {
            NvSciBufAttrListFree(m_elems[i].bufAttrList);
        }
    }
}

SIPLStatus CPoolManager::Init(void)
{
    LOG_DBG("Pool Init.\n");

    /* Query number of consumers */
    auto sciErr = NvSciStreamBlockConsumerCountGet(m_handle, &m_numConsumers);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query number of consumers");
    LOG_MSG("Pool: Consumer count is %u\n", m_numConsumers);

    return NVSIPL_STATUS_OK;
}

EventStatus CPoolManager::HandleEvents(void)
{
    NvSciStreamEventType event;
    SIPLStatus status = NVSIPL_STATUS_OK;
    NvSciError sciStatus;

    auto sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &event);
    if (NvSciError_Success != sciErr) {
        if (NvSciError_Timeout == sciErr) {
            LOG_WARN("Pool: Event query, timed out.\n");
            return EVENT_STATUS_TIMED_OUT;
        }
        LOG_ERR("Pool: Event query, failed with error 0x%x", sciErr);
        return EVENT_STATUS_ERROR;
    }

    switch (event) {
        case NvSciStreamEventType_Connected:
            /* handleElemSupport(); */
            printf("HandleEvent, received NvSciStreamEventType_Connected.\n");
            break;
        /* Process all element support from producer and consumer(s) */
        case NvSciStreamEventType_Elements:
            printf("HandleEvent, received NvSciStreamEventType_Elements.\n");
            status = HandlePoolBufferSetup();
            break;
        case NvSciStreamEventType_PacketStatus:
            if (++m_numPacketReady < m_numPackets) {
                break;
            }

            printf("Pool: Received all the PacketStatus events.\n");
            status = HandlePacketsStatus();
            break;
        case NvSciStreamEventType_Error:
            sciErr = NvSciStreamBlockErrorGet(m_handle, &sciStatus);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed to query the error event code 0x%x\n", sciErr);
            } else {
                LOG_ERR("Pool: Received error event: 0x%x\n", sciStatus);
            }
            status = NVSIPL_STATUS_ERROR;
            break;
        case NvSciStreamEventType_Disconnected:
            if (!m_elementsDone) {
                LOG_WARN("Pool: Disconnect before element support\n");
            } else if (!m_packetsDone) {
                LOG_WARN("Pool: Disconnect before packet setup\n");
            }
            status = NVSIPL_STATUS_ERROR;
            break;
        /* All setup complete. Transition to runtime phase */
        case NvSciStreamEventType_SetupComplete:
            printf("Pool: Setup completed\n");
            return EVENT_STATUS_COMPLETE;

        default:
            printf("Pool: Received unknown event 0x%x\n", event);
            status = NVSIPL_STATUS_ERROR;
            break;
    }
    return (status == NVSIPL_STATUS_OK) ? EVENT_STATUS_OK : EVENT_STATUS_ERROR;
}

SIPLStatus CPoolManager::HandlePoolBufferSetup(void)
{
    /* Query producer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Producer, &m_numProdElem);
        printf("m_numProdElem=%d,\n",m_numProdElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query producer element count");

    /* Query consumer element count */
    sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Consumer, &m_numConsElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query consumer element count");

    /* Query all producer elements */
    for (uint32_t i = 0U; i < m_numProdElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Producer, i, &m_prodElems[i].userName,
                                                &m_prodElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to query producer element %u\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Query all consumer elements */
    for (uint32_t i = 0U; i < m_numConsElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Consumer, i, &m_consElems[i].userName,
                                                &m_consElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to query consumer element %d\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Indicate that all element information has been imported */
    m_elementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");

    uint32_t numElem = 0, p, c, e, i;
    ElemAttr elem[MAX_NUM_ELEMENTS];
    std::vector<NvSciBufAttrList> unreconciled;
    for (p = 0; p < m_numProdElem; ++p) {
        ElemAttr *prodElem = &m_prodElems[p];
        unreconciled.push_back(prodElem->bufAttrList);
        for (c = 0; c < m_numConsElem; ++c) {
            ElemAttr *consElem = &m_consElems[c];
            /* If requested element types match, combine the entries */
            if (prodElem->userName == consElem->userName) {
                unreconciled.push_back(consElem->bufAttrList);
                /* Found a match for this producer element so move on */
                break;
            } /* if match */
        }     /* for all requested consumer elements */

        ElemAttr *poolElem = &m_elems[numElem++];
        poolElem->userName = prodElem->userName;
        poolElem->bufAttrList = nullptr;

        /* Combine and reconcile the attribute lists */
        NvSciBufAttrList conflicts = nullptr;
        sciErr =
            NvSciBufAttrListReconcile(unreconciled.data(), unreconciled.size(), &poolElem->bufAttrList, &conflicts);
        if (nullptr != conflicts) {
            NvSciBufAttrListFree(conflicts);
        }
        unreconciled.clear();

        /* Abort on error */
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed to reconcile element 0x%x attrs (0x%x)\n", poolElem->userName, sciErr);
            return NVSIPL_STATUS_ERROR;
        }
    } /* for all requested producer elements */

    /* Should be at least one element */
    if (0 == numElem) {
        LOG_ERR("Pool: Didn't find any common elements\n");
        return NVSIPL_STATUS_ERROR;
    }

    /* The requested attribute lists are no longer needed, so discard them */
    for (p = 0; p < m_numProdElem; ++p) {
        ElemAttr *prodElem = &m_prodElems[p];
        if (nullptr != prodElem->bufAttrList) {
            NvSciBufAttrListFree(prodElem->bufAttrList);
            prodElem->bufAttrList = nullptr;
        }
    }
    for (c = 0; c < m_numConsElem; ++c) {
        ElemAttr *consElem = &m_consElems[c];
        if (nullptr != consElem->bufAttrList) {
            NvSciBufAttrListFree(consElem->bufAttrList);
            consElem->bufAttrList = nullptr;
        }
    }

    /* Inform the stream of the chosen elements */
    for (e = 0; e < numElem; ++e) {
        ElemAttr *poolElem = &m_elems[e];
        sciErr = NvSciStreamBlockElementAttrSet(m_handle, poolElem->userName, poolElem->bufAttrList);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to send element %u info\n", sciErr, e);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element export");

    /*
     * Create and send all the packets and their buffers
     * Note: Packets and buffers are not guaranteed to be received by
     *       producer and consumer in the same order sent, nor are the
     *       status messages sent back guaranteed to preserve ordering.
     *       This is one reason why an event driven model is more robust.
     */
    for (i = 0; i < m_numPackets; ++i) {
        /*Our pool implementation doesn't need to save any packet-specific
         *   data, but we do need to provide unique cookies, so we just
         *   use the pointer to the location we save the handle.
         */
        NvSciStreamCookie cookie = (NvSciStreamCookie)&m_packetHandles[i];
        sciErr = NvSciStreamPoolPacketCreate(m_handle, cookie, &m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to create packet %d\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }

        /* Create buffers for the packet */
        for (e = 0; e < numElem; ++e) {
            /* Allocate a buffer object */
            NvSciBufObj obj = nullptr;
            sciErr = NvSciBufObjAlloc(m_elems[e].bufAttrList, &obj);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to allocate buffer %u of packet %u\n", sciErr, e, i);
                return NVSIPL_STATUS_ERROR;
            }
            /* Insert the buffer in the packet */
            sciErr = NvSciStreamPoolPacketInsertBuffer(m_handle, m_packetHandles[i], e, obj);
            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
            obj = nullptr;
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to insert buffer %u of packet %u\n", sciErr, e, i);
                return NVSIPL_STATUS_ERROR;
            }
        }
        /* Indicate packet setup is complete */
        sciErr = NvSciStreamPoolPacketComplete(m_handle, m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to complete packet %u setup\n", sciErr, i);
            return NVSIPL_STATUS_ERROR;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the status has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet export");

    /* Once all packets are set up, no longer need to keep the attributes */
    for (e = 0; e < numElem; ++e) {
        ElemAttr *poolElem = &elem[e];
        if (nullptr != poolElem->bufAttrList) {
            NvSciBufAttrListFree(poolElem->bufAttrList);
            poolElem->bufAttrList = nullptr;
        }
    }

    return NVSIPL_STATUS_OK;
}

/* Check packet status */
SIPLStatus CPoolManager::HandlePacketsStatus(void)
{
    bool packetFailure = false;
    NvSciError sciErr;

    /* Check each packet */
    for (uint32_t p = 0; p < m_numPackets; ++p) {
        /* Check packet acceptance */
        bool accept;
        sciErr = NvSciStreamPoolPacketStatusAcceptGet(m_handle, m_packetHandles[p], &accept);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's acceptance-statue\n", sciErr, p);
            return NVSIPL_STATUS_ERROR;
        }
        if (accept) {
            continue;
        }

        /* On rejection, query and report details */
        packetFailure = true;
        NvSciError status;

        /* Check packet status from producer */
        sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Producer, 0U,
                                                     &status);
        if (NvSciError_Success != sciErr) {
            LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's statue from producer\n", sciErr, p);
            return NVSIPL_STATUS_ERROR;
        }
        if (status != NvSciError_Success) {
            LOG_ERR("Pool: Producer rejected packet %u with error 0x%x\n", p, status);
        }

        /* Check packet status from consumers */
        for (uint32_t c = 0; c < m_numConsumers; ++c) {
            sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Consumer, c,
                                                         &status);
            if (NvSciError_Success != sciErr) {
                LOG_ERR("Pool: Failed (0x%x) to retrieve packet %u's statue from consumer %u\n", sciErr, p, c);
                return NVSIPL_STATUS_ERROR;
            }
            if (status != NvSciError_Success) {
                LOG_ERR("Pool: Consumer %u rejected packet %d with error 0x%x\n", c, p, status);
            }
        }
    }

    /* Indicate that status for all packets has been received. */
    m_packetsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet import");

    return packetFailure ? NVSIPL_STATUS_ERROR : NVSIPL_STATUS_OK;
}
