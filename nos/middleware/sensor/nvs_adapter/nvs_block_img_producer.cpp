#include "sensor/nvs_adapter/nvs_block_img_producer.h"

namespace hozon {
namespace netaos {
namespace nv { 

NVSBlockImgProducer::NVSBlockImgProducer() {
}

NVSBlockImgProducer::~NVSBlockImgProducer() {

}

int32_t NVSBlockImgProducer::Create(NvSciStreamBlock pool, const std::string& endpoint_info) {
    name = "Producer";

    NvSciError err = NvSciStreamProducerCreate(pool, &block);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to create producer, ret " << LogHexNvErr(err); 
        return -1;
    }

    err = NvSciStreamBlockUserInfoSet(block,
                                        ENDINFO_NAME_PROC,
                                        endpoint_info.size(), endpoint_info.data());
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to setup producer info, ret " << LogHexNvErr(err);
        return -1;
    }

    RegIntoEventService();

    return 0;
}

void NVSBlockImgProducer::DeleteBlock() {
    NvSciStreamBlockDelete(block);

    for (auto& packet : _packets) {
        NvSciBufObjFree(packet->nv_sci_buf);
    }

    NvSciBufAttrListFree(_source_attr);
    NvSciSyncObjFree(_signal_obj);

    for (uint32_t i = 0; i < MAX_CONSUMERS; ++i) {
        if (_waiter_obj[i] != nullptr) {
            NvSciSyncObjFree(_waiter_obj[i]);
        }
    }

    NvSciSyncCpuWaitContextFree(_cpu_wait_context);
}

int32_t NVSBlockImgProducer::OnConnected() {
    int32_t ret = StreamInit();
    if (ret < 0) {
        return -1;
    }

    ret = ProducerInit();
    if (ret < 0) {
        return -2;
    }

    ret = ProducerElemSupport();
    if (ret < 0) {
        return -3;
    }

    ret = ProducerSyncSupport();
    if (ret < 0) {
        return -4;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnElements() {
    NvSciError err;

    /*
     * This application does not need to query the element count, because we
     *   know it is always 1. But we do so anyways to show how it is done.
     */
    uint32_t count;
    err = NvSciStreamBlockElementCountGet(block,
                                          NvSciStreamBlockType_Pool,
                                          &count);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to query element count, ret " << LogHexNvErr(err);
        return -1;
    }
    NVS_LOG_INFO << "Producer received element count " << count;

    /*
     * Query element type and attributes.
     *   For this simple use case, there is only one type and the attribute
     *   list is not needed, so we could skip this call. We do it only to
     *   illustrate how it is done.
     */
    uint32_t type;
    NvSciBufAttrList bufAttr;
    err = NvSciStreamBlockElementAttrGet(block,
                                         NvSciStreamBlockType_Pool, 0U,
                                         &type, &bufAttr);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to query element attr, ret " << LogHexNvErr(err);
        return -1;
    }
    NVS_LOG_INFO << "Producer received elment type " << log::loghex(type);

    /* Don't need to keep attribute list */
    NvSciBufAttrListFree(bufAttr);

    /* Indicate that element import is complete */
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_ElementImport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to complete element import, ret " << LogHexNvErr(err); 
        return -1;
    }

    /* Set waiter attributes for the asynchronous element. */
    err = NvSciStreamBlockElementWaiterAttrSet(block, 0U, _waiter_attr);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to send waiter attrs, ret " << LogHexNvErr(err);
        return -1;
    }

    /* Once sent, the waiting attributes are no longer needed */
    NvSciSyncAttrListFree(_waiter_attr);
    _waiter_attr = NULL;

    /* Indicate that waiter attribute export is done. */
    err = NvSciStreamBlockSetupStatusSet(block,
                                         NvSciStreamSetup_WaiterAttrExport,
                                         true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to complete waiter attr export, ret " << LogHexNvErr(err);
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnPacketCreate() {
    NvSciError err;

    /* Retrieve handle for packet pending creation */
    NvSciStreamPacket handle;
    err = NvSciStreamBlockPacketNewHandleGet(block, &handle);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to retrieve handle for the new packet, ret " << LogHexNvErr(err);
        return -1;
    }

    // /* Make sure there is room for more packets */
    // if (MAX_PACKETS <= _num_packet) {
    //     printf("Producer exceeded max packets\n");
    //     err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
    //                                           handle,
    //                                           NvSciStreamCookie_Invalid,
    //                                           NvSciError_Overflow);
    //     if (NvSciError_Success != err) {
    //         printf("Producer failed (%x) to inform pool of packet status\n",
    //                err);
    //     }
    //     return 0;
    // }

    /* Allocate the next entry in the array for the new packet. */
    NvSciError sciErr;
    std::shared_ptr<ImageProducerPacket> packet(new ImageProducerPacket);
    _packets.emplace_back(packet);
    packet->handle = handle;

    /* Allocate a source buffer for the NvMedia operations */
    sciErr = NvSciBufObjAlloc(_source_attr, &packet->nv_sci_buf);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Producer fail to allocate source buffer, ret " << LogHexNvErr(sciErr);
        return -1;
    } 

    // /* Register the source buffer */
    // nvmErr = NvMedia2DRegisterNvSciBufObj(prodData->nvm2d, packet->srcBuf);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to register sources buffer.\n", nvmErr);
    //     return 0;
    // }

    // /* Get CPU pointer */
    // uint8_t *cpu_ptr = NULL;
    // sciErr = NvSciBufObjGetCpuPtr(packet->srcBuf, (void **)&cpu_ptr);
    // if (NvSciError_Success != sciErr) {
    //         printf("Producer failed (%x) to get cpu pointer\n", sciErr);
    // } else {
    //     /* Get width, height and pitch attributes */
    //     NvSciBufAttrKeyValuePair attr[] =
    //     {
    //         { NvSciBufImageAttrKey_PlaneWidth, NULL,0},
    //         { NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
    //         { NvSciBufImageAttrKey_PlanePitch, NULL, 0}
    //     };

    //     sciErr = NvSciBufAttrListGetAttrs(prodData->sourceAttr, attr,
    //                                 sizeof(attr)/sizeof(NvSciBufAttrKeyValuePair));
    //     if (NvSciError_Success != sciErr) {
    //         printf("Producer failed (%x) to get attributes\n", sciErr);
    //     }

    //     if (NvSciError_Success == sciErr) {
    //         uint32_t width  = *(uint32_t *)attr[0].value;
    //         uint32_t height = *(uint32_t *)attr[1].value;
    //         uint32_t pitch  = *(uint32_t *)attr[2].value;
    //     //    printf("width: %d, height: %d, pitch: %d, p: %d\n", width, height, pitch, p); 
    //         uint32_t nWidth = 4U;
    //         uint8_t* srcPtr = cpu_ptr;

    //         for (uint32_t y = 0U; y < height; y++) {
    //             for (uint32_t x = 0U; x < width * nWidth; x++) {
    //                 srcPtr[x] = (p + ((x % 32)+ (y % 32) )) % (1 << 8);
    //             }
    //             srcPtr += pitch;
    //         }
    //         printf("Val: %d%d%d%d\n", srcPtr[0], srcPtr[1], srcPtr[2], srcPtr[3]);
    //     }
    
    // }

    // /* Inform pool of failure */
    // if (NvSciError_Success != sciErr) {
    //     err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
    //                                           handle,
    //                                           NvSciStreamCookie_Invalid,
    //                                           sciErr);
    //     if (NvSciError_Success != err) {
    //         printf("Producer failed (%x) to inform pool of packet status\n",
    //                err);
    //     }
    //     return 0;
    // }


    /* Handle mapping of a packet buffer */
    sciErr = NvSciError_Success;

    /* Retrieve all buffers and map into application
     *   This use case has only 1 element.
     */
    NvSciBufObj bufObj;
    err = NvSciStreamBlockPacketBufferGet(block,
                                          handle,
                                          0U,
                                          &bufObj);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to retrieve buffe, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Save buffer object */
    packet->nv_sci_buf = bufObj;

    // /* Register the data buffer */
    // nvmErr = NvMedia2DRegisterNvSciBufObj(prodData->nvm2d, packet->dataBuf);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to register data buffer.\n", nvmErr);
    //     return 0;
    // }

    // /* Get datda buffer attributes */
    // NvSciBufAttrList databufAttr = NULL;
    // sciErr = NvSciBufObjGetAttrList(packet->dataBuf, &databufAttr);
    // if (NvSciError_Success != sciErr) {
    //     printf("Producer failed (%x) to get attribute list\n", sciErr);
    //     return 0;
    // }

    // /* Get CPU pointer */
    // uint8_t *cpu_ptr2 = NULL;
    // sciErr = NvSciBufObjGetCpuPtr(packet->dataBuf, (void **)&cpu_ptr2);
    // if (NvSciError_Success != sciErr) {
    //     printf("Producer failed (%x) to get cpu pointer\n", sciErr);
    //     return 0;
    // }

    // /* Get width, height and pitch attributes */
    // NvSciBufAttrKeyValuePair attr2[] =
    // {
    //     { NvSciBufImageAttrKey_PlaneHeight, NULL, 0},
    //     { NvSciBufImageAttrKey_PlanePitch, NULL, 0}
    // };

    // sciErr = NvSciBufAttrListGetAttrs(databufAttr, attr2,
    //                                   sizeof(attr2)/sizeof(NvSciBufAttrKeyValuePair));
    // if (NvSciError_Success != sciErr) {
    //     printf("Producer failed (%x) to get attributes\n", sciErr);
    //     return 0;
    // } else {
    //     uint32_t height2 = *(uint32_t *)attr2[0].value;
    //     uint32_t pitch2  = *(uint32_t *)attr2[1].value;
    //     uint8_t* data_ptr = cpu_ptr2;

    //     (void)memset(data_ptr, 0, pitch2 * height2);
    // }

    // /* Inform pool of succes or failure */
    // if (NvSciError_Success != sciErr) {
    //     err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
    //                                           packet->handle,
    //                                           NvSciStreamCookie_Invalid,
    //                                           sciErr);
    // } else {
    //     err = NvSciStreamBlockPacketStatusSet(prodData->common.block,
    //                                           packet->handle,
    //                                           (NvSciStreamCookie)packet,
    //                                           NvSciError_Success);
    // }
    // if (NvSciError_Success != err) {
    //     printf("Producer failed (%x) to inform pool of packet status\n",
    //            err);
    //     return 0;
    // }

    err = NvSciStreamBlockPacketStatusSet(block,
                                            packet->handle,
                                            (NvSciStreamCookie)packet.get(),
                                            NvSciError_Success);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to retrieve buffer, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnPacketsComplete() {
   NvSciError err = NvSciStreamBlockSetupStatusSet(block,
                                            NvSciStreamSetup_PacketImport,
                                            true);
    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Producer fail to complete packet import, ret " << LogHexNvErr(err);
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnWaiterAttr() {
    NvSciError        sciErr;
    // NvMediaStatus     nvmErr;

    /* Process waiter attrs from all elements.
     * This use case has only one element.
     */
    NvSciSyncAttrList waiterAttr = NULL;
    sciErr = NvSciStreamBlockElementWaiterAttrGet(block,
                                                  0U, &waiterAttr);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to query waiter attr, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    if (NULL == waiterAttr) {
        NVS_LOG_CRITICAL << "Producer received NULL waiter attr for data elem";
        return -1;
    }

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(block,
                                            NvSciStreamSetup_WaiterAttrImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to complete waiter attr import, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /*
     * Merge and reconcile consumer sync attrs with ours.
     * Note: Many producers would only require their signaler attributes
     *       and the consumer waiter attributes. As noted above, we also
     *       add in attributes to allow us to CPU wait for the last fence
     *       generated by its sync object.
     */
    NvSciSyncAttrList unreconciled[2] = {
        _signal_attr,
        waiterAttr
    };
    NvSciSyncAttrList reconciled = NULL;
    NvSciSyncAttrList conflicts = NULL;
    sciErr = NvSciSyncAttrListReconcile(unreconciled, 2,
                                        &reconciled, &conflicts);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to reconcile sync attributes, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Allocate sync object */
    sciErr = NvSciSyncObjAlloc(reconciled, &_signal_obj);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to allocate sync object, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Free the attribute lists */
    NvSciSyncAttrListFree(_signal_attr);
    _signal_attr = NULL;
    NvSciSyncAttrListFree(waiterAttr);
    NvSciSyncAttrListFree(reconciled);

    // /* Register sync object with NvMedia */
    // nvmErr = NvMedia2DRegisterNvSciSyncObj(prodData->nvm2d,
    //                                        NVMEDIA_EOFSYNCOBJ,
    //                                        prodData->signalObj);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to register signal sync object\n",
    //            nvmErr);
    //     return 0;
    // }

    /* Send the sync object for each element */
    sciErr = NvSciStreamBlockElementSignalObjSet(block,
                                                 0U,
                                                 _signal_obj);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to send sync object, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Indicate that sync object export is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(block,
                                            NvSciStreamSetup_SignalObjExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to complete signal obj export, ret " << LogHexNvErr(sciErr);

        return -1;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnSignalObj() {
    // NvMediaStatus     nvmErr;

    NvSciError        sciErr;

    /* Query sync objects for each element
     * from all consumers.
     */
    for (uint32_t c = 0U; c < _num_consumers; c++) {
        NvSciSyncObj waiterObj = NULL;
        sciErr = NvSciStreamBlockElementSignalObjGet(
                    block,
                    c, 0U, &waiterObj);
        if (NvSciError_Success != sciErr) {
            NVS_LOG_CRITICAL << "Fail to  query sync obj from consumer " << c << ", ret " << LogHexNvErr(sciErr);
            return -1;
        }

        /* Save object */
        _waiter_obj[c] = waiterObj;

        // /* If the waiter sync obj is NULL,
        //  * it means this element is ready to use when received.
        //  */
        // if (NULL != waiterObj) {
        //     /* Register sync object with NvMedia */
        //     nvmErr = NvMedia2DRegisterNvSciSyncObj(prodData->nvm2d,
        //                                            NVMEDIA_PRESYNCOBJ,
        //                                            waiterObj);
        //     if (NVMEDIA_STATUS_OK != nvmErr) {
        //         printf("Producer failed (%x) to register waiter sync object\n",
        //                nvmErr);
        //         return 0;
        //     }
        // }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(block,
                                            NvSciStreamSetup_SignalObjImport,
                                            true);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to complete signal obj import, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    return 0;
}

int32_t NVSBlockImgProducer::OnPacketReady() {
    NvSciError        sciErr;
    // NvMediaStatus     nvmErr;

    /* Obtain packet for the new payload */
    NvSciStreamCookie cookie;
    sciErr = NvSciStreamProducerPacketGet(block,
                                          &cookie);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to obtain packet for payload, ret " << LogHexNvErr(sciErr);
        return -1;
    }
    ImageProducerPacket* packet = (ImageProducerPacket*)cookie;

    // NvMedia2DComposeParameters params;
    // nvmErr = NvMedia2DGetComposeParameters(prodData->nvm2d,
    //                                            &params);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to get compose parameters\n",
    //             nvmErr);
    //     return 0;
    // }

    // nvmErr = NvMedia2DSetNvSciSyncObjforEOF(prodData->nvm2d,
    //                                         params,
    //                                         prodData->signalObj);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to set EOF sync object\n", nvmErr);
    //     return 0;
    // }

    /* Query fences for this element from each consumer */
    for (uint32_t i = 0U; i < _num_consumers; ++i) {
        /* If the received waiter obj if NULL,
         * the consumer is done using this element,
         * skip waiting on pre-fence.
         */
        if (NULL == _waiter_obj[i]) {
            continue;
        }

        NvSciSyncFence prefence = NvSciSyncFenceInitializer;
        sciErr = NvSciStreamBlockPacketFenceGet(
                    block,
                    packet->handle,
                    i, 0U,
                    &prefence);
        if (NvSciError_Success != sciErr) {
            NVS_LOG_CRITICAL << "Failed to query fence from consumer " << i << ", ret " << LogHexNvErr(sciErr);
            return -1;
        }

        sciErr = NvSciSyncFenceWait(&prefence, _cpu_wait_context, 0xFFFFFFFF);
        if (NvSciError_Success != sciErr) {
            NVS_LOG_CRITICAL << "Failed to wait for consumer, ret " << LogHexNvErr(sciErr);
            return -1;
        }


        // /* Instruct NvMedia to wait for each of the consumer fences */
        // nvmErr = NvMedia2DInsertPreNvSciSyncFence(prodData->nvm2d,
        //                                           params,
        //                                           &prefence);
        NvSciSyncFenceClear(&prefence);

        // if (NVMEDIA_STATUS_OK != nvmErr) {
        //     printf("Producer failed (%x) to wait for prefence %d\n",
        //            nvmErr, i);
        //     return 0;
        // }
    }

    // uint32_t index = 0;
    // nvmErr = NvMedia2DSetSrcNvSciBufObj(prodData->nvm2d, params, index, packet->srcBuf);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to set source buf\n",
    //            nvmErr);
    //     return 0;
    // }

    // nvmErr = NvMedia2DSetDstNvSciBufObj(prodData->nvm2d, params, packet->dataBuf);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to set source buf\n",
    //            nvmErr);
    //     return 0;
    // }

    // NvMedia2DComposeResult result;
    // nvmErr = NvMedia2DCompose(prodData->nvm2d, params, &result);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to compose\n",
    //            nvmErr);
    //     return 0;
    // }

    // /* Instruct NvMedia to signal the post fence */
    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    sciErr = NvSciSyncObjGenerateFence(_signal_obj, &postfence);
    if (sciErr != NvSciError_Success) {
        NVS_LOG_CRITICAL << "Failed to generate fence, ret " << LogHexNvErr(sciErr);
        return -1;
    }
    // nvmErr = NvMedia2DGetEOFNvSciSyncFence(prodData->nvm2d,
    //                                        &result,
    //                                        &postfence);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to signal postfence\n", nvmErr);
    //     return 0;
    // }

    /* Update postfence for this element */
    sciErr = NvSciStreamBlockPacketFenceSet(block,
                                            packet->handle,
                                            0U,
                                            &postfence);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to set fence, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Send the new payload to the consumer(s) */
    sciErr = NvSciStreamProducerPacketPresent(block,
                                              packet->handle);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to present packet ret " << LogHexNvErr(sciErr);
        return -1;
    }
    NvSciSyncObjSignal(_signal_obj);


    /* If counter has reached the limit, indicate finished */
    if (++(_counter) == 32) {
        /* Make sure all operations have been completed
         * before resource cleanup.
         */
        sciErr =  NvSciSyncFenceWait(&postfence,
                            _cpu_wait_context,
                            0xFFFFFFFF);
        if (NvSciError_Success != sciErr) {
            NVS_LOG_CRITICAL << "Fail to wait for all operations done, ret " << LogHexNvErr(sciErr);
            return -1;
        }

        NVS_LOG_INFO << "Producer finished sending " << _counter << " payloads";
    }

    NvSciSyncFenceClear(&postfence);

    return 1;
}

int32_t NVSBlockImgProducer::StreamInit() {
    /* Query number of consumers */
    NvSciError err =
        NvSciStreamBlockConsumerCountGet(block,
                                         &_num_consumers);

    if (NvSciError_Success != err) {
        NVS_LOG_CRITICAL << "Fail to query the number of consumers " << LogHexNvErr(err);
        return -1;
    }
    NVS_LOG_INFO << "Query consumer numbers " << _num_consumers;

    for (uint32_t i = 0U; i < _num_consumers; i++) {
        uint32_t size = INFO_SIZE;
        char info[INFO_SIZE];
        err = NvSciStreamBlockUserInfoGet(
                block,
                NvSciStreamBlockType_Consumer, i,
                ENDINFO_NAME_PROC,
                &size, &info);
        if (NvSciError_Success == err) {
            NVS_LOG_INFO << "Consumer " << i << " info " << std::string(info);
        } 
        else if (NvSciError_StreamInfoNotProvided == err) {
            NVS_LOG_INFO << "Consumer " << i << " info not provided.";
        } 
        else {
            NVS_LOG_CRITICAL << "Consumer " << i << ", fail to query info " << LogHexNvErr(err);
            return -1;
        }
    }

    return 0;
}

int32_t NVSBlockImgProducer::ProducerInit() {
    return 0; // do nothing
}

int32_t NVSBlockImgProducer::ProducerElemSupport() {
    NvSciError sciErr;
    uint32_t bufName = ELEMENT_NAME_IMAGE;

    NvSciBufAttrList       bufAttr = NULL;
    /* Create unreconciled attribute list for NvMedia buffers */
    sciErr = NvSciBufAttrListCreate(NVSHelper::GetInstance().sci_buf_module, &bufAttr);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to create buffer attribute list, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Add read/write permission to attribute list */
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrKeyValuePair  bufKeyVal =
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) };
    sciErr = NvSciBufAttrListSetAttrs(bufAttr, &bufKeyVal, 1);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to set source permission, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Get Nvmedia surface type for A8R8G8B8 buffers */
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValColorFmt colorFmt = NvSciColor_A8R8G8B8;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    uint32_t planeCount = 1;
    NvSciBufAttrKeyValuePair bufFormatKeyVal[4] =
        {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufImageAttrKey_PlaneColorFormat, &colorFmt, sizeof(colorFmt) },
            { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
            { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
        };

    sciErr = NvSciBufAttrListSetAttrs(bufAttr, bufFormatKeyVal, 4);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to set buffer format attributes, ret " << LogHexNvErr(sciErr);
        return 0;
    }

    /* Set NvMedia surface allocation attributes */
    bool enableCpuCache = true;
    bool needCpuAccess = true;
    uint64_t topPadding = 0;
    uint64_t bottomPadding = 0;
    NvSciBufAttrValColorStd colorStd = NvSciColorStd_REC601_ER;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    uint32_t const WIDTH  = 1920;
    uint32_t const HEIGHT = 1080;
    NvSciBufAttrKeyValuePair bufAllocKeyVal[8] =
        {
            { NvSciBufImageAttrKey_PlaneWidth, &WIDTH, sizeof(WIDTH) },
            { NvSciBufImageAttrKey_PlaneHeight, &HEIGHT, sizeof(HEIGHT) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &enableCpuCache, sizeof(enableCpuCache) },
            { NvSciBufImageAttrKey_TopPadding, &topPadding, sizeof(topPadding) },
            { NvSciBufImageAttrKey_BottomPadding, &bottomPadding, sizeof(bottomPadding) },
            { NvSciBufImageAttrKey_PlaneColorStd, &colorStd, sizeof(colorStd) },
            { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(scanType) }
        };

    sciErr = NvSciBufAttrListSetAttrs(bufAttr, bufAllocKeyVal, 8);
    if (NvSciError_Success != sciErr) {
        printf("Producer failed (%x) to set buffer alloc attributes\n", sciErr);
        return 0;
    }

    /* Inform stream of the attributes */
    sciErr = NvSciStreamBlockElementAttrSet(block, bufName, bufAttr);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to send element attribute, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(block,
                                            NvSciStreamSetup_ElementExport,
                                            true);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to complete element export, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Also reconcile and save the attributes for source buffer creation */
    NvSciBufAttrList conflicts = NULL;
    sciErr = NvSciBufAttrListReconcile(&bufAttr, 1, &_source_attr, &conflicts);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to reconcile source attributes, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    /* Clean up */
    NvSciBufAttrListFree(bufAttr);
    if (NULL != conflicts) {
        NvSciBufAttrListFree(conflicts);
    }

    return 0;
}

int32_t NVSBlockImgProducer::ProducerSyncSupport() {
    NvSciError       sciErr;
    // NvMediaStatus    nvmErr;

    /*
     * Create sync attribute list for signaling.
     *   This will be saved until we receive the consumer's attributes
     */
    sciErr = NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &_signal_attr);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to allocate signal sync attrs, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    uint8_t cpu_sync = 1;
    NvSciSyncAccessPerm cpu_perm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrKeyValuePair cpu_key_vals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &cpu_sync, sizeof(cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &cpu_perm, sizeof(cpu_perm) }
    };

    sciErr = NvSciSyncAttrListSetAttrs(_signal_attr, cpu_key_vals, 2);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to fill cpu signal sync attrs, ret " << LogHexNvErr(sciErr);
        return -1;
    }


    /* Create sync attribute list for waiting. */
    sciErr = NvSciSyncAttrListCreate(NVSHelper::GetInstance().sci_sync_module, &_waiter_attr);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to allocate signal sync attrs, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    uint8_t waiter_cpu_sync = 1;
    NvSciSyncAccessPerm waiter_cpu_perm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair waiter_cpu_key_vals[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &waiter_cpu_sync, sizeof(waiter_cpu_sync) },
        { NvSciSyncAttrKey_RequiredPerm,  &waiter_cpu_perm, sizeof(waiter_cpu_perm) }
    };

    sciErr = NvSciSyncAttrListSetAttrs(_waiter_attr, waiter_cpu_key_vals, 2);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to fill cpu waiter sync attrs, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    // /* Have NvMedia fill the waiting attribute list */
    // nvmErr = NvMedia2DFillNvSciSyncAttrList(prodData->nvm2d,
    //                                         prodData->waiterAttr,
    //                                         NVMEDIA_WAITER);
    // if (NVMEDIA_STATUS_OK != nvmErr) {
    //     printf("Producer failed (%x) to fill waiter sync attrs\n", nvmErr);
    //     return 0;
    // }


    /* Create a context for CPU waiting */
    sciErr = NvSciSyncCpuWaitContextAlloc(NVSHelper::GetInstance().sci_sync_module, &_cpu_wait_context);
    if (NvSciError_Success != sciErr) {
        NVS_LOG_CRITICAL << "Fail to fill to create CPU wait context, ret " << LogHexNvErr(sciErr);
        return -1;
    }

    return 0;
}

}
}
}