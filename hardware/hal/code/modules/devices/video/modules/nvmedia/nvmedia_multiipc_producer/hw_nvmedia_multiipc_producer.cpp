#include "hw_nvmedia_compile.h"

#if (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIIPC_PRODUCER)
#include "hw_nvmedia_common_impl.h"
#include "hw_nvmedia_multiipc_producer_impl.h"
#include "hw_nvmedia_multiipc_consumer_enc_impl.h"

extern "C" struct hw_nvmedia_multiipc_consumer_enc_module_t HAL_MODULE_INFO_SYM_ENC;
extern "C" struct hw_nvmedia_multiipc_producer_module_t HAL_MODULE_INFO_SYM;
struct hw_nvmedia_multiipc_producer_module_t HAL_MODULE_INFO_SYM =
{
	.common = {
		.tag = HARDWARE_MODULE_TAG,
		.hal_api_version = HARDWARE_HAL_API_VERSION,
		.global_devicetype_version = HW_GLOBAL_DEVTYPE_VERSION,
		.devicetype = HW_DEVICETYPE_VIDEO,
        .module_id = HW_VIDEO_MODULEID_NVMEDIA_MULTIGROUP,
		.devtype_magic = hw_global_devtype_magic_get(HW_DEVICETYPE_VIDEO),
		.device_moduleid_version = HW_DEVICE_MODULEID_VERSION,
		.module_api_version = HW_MODULE_API_VERSION,
            .devmoduleid_magic = hw_video_moduleid_magic_get(HW_VIDEO_MODULEID_NVMEDIA_MULTIGROUP),
		.description = "ipc_producer block " HW_NVMEDIA_VERSION,
		.privdata = {
			.dso = NULL,
			.pvoid = NULL,
		},
		.privapi = {
			.init = hw_module_privapi_init,
			.check_device_api_version = hw_module_privapi_check_device_api_version,
			.trigger_check_unmasklogleft = hw_module_privapi_trigger_check_unmasklogleft,
			.device_get = hw_module_privapi_device_get,
			.device_put = hw_module_privapi_device_put,
		},
	},
	.nvmedia = {
		.driveosversiondesc = "6.0.6",
	},
};

extern s32 hw_module_private_early_attach();
static HWNvmediaMultiIpcProducerEnv		_env;
static hw_module_t* _psubmodule=nullptr;
// static hw_device_t* _psubdevice=nullptr;
static std::thread  _psubthread;
static bool _mainend = false;

s32 hw_module_privapi_init(void** io_ppvoid)
{
	if (_env.binit == 0)
	{
		HW_NVMEDIA_LOG_UNMASK("Run init of module[%s]nvmedia-driveosversiondesc[%s]!\r\n", HAL_MODULE_INFO_SYM.common.description,
			HAL_MODULE_INFO_SYM.nvmedia.driveosversiondesc);
		/*
		* Set all of the global .data content the init value.
		*/

		/*
		* For future reserved.
		*/
		*io_ppvoid = NULL;

		_env.binit = 1;
		return 0;
	}
	return -1;
}

s32 hw_module_privapi_check_device_api_version(u32 i_device_api_version)
{
	_env.device_api_version = i_device_api_version;
	/*
	* The module currently only support HARDWARE_DEVICE_API_VERSION(0, 1).
	*/
	if (i_device_api_version == HW_MAKEV_DEVICE_API_VERSION(0, 1))
	{
		_env.bsupport_device_api_version = 1;
		HW_NVMEDIA_LOG_UNMASK("The current module support the device_api_version[%x]!\r\n", _env.device_api_version);
		return 0;
	}
	else
	{
		_env.bsupport_device_api_version = 0;
		HW_NVMEDIA_LOG_UNMASK("The current module does not support the device_api_version[%x]!\r\n", _env.device_api_version);
		return -1;
	}
}

s32 hw_module_privapi_trigger_check_unmasklogleft()
{
	/*
	* For future use.
	*/
	return 0;
}

s32 hw_module_privapi_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** io_ppdevice)
{
	/*
	* Inner use only. Only for nvmedia internal api currently.
	*/
	hw_ret_s32 ret;
	/*
	* First need to check whether support the current device api version. We already check it
	* when call hw_module_get in the previous step.
	*/
	if (HW_UNLIKELY(_env.bsupport_device_api_version == 0))
	{
		HW_NVMEDIA_LOG_UNMASK("bsupport_device_api_version is 0, so cannot get device!\r\n");
		return -1;
	}
	struct hw_video_t* pvideo = (struct hw_video_t*)malloc(sizeof(struct hw_video_t));
	pvideo->common.tag = HARDWARE_DEVICE_TAG;
	pvideo->common.device_api_version = _env.device_api_version;
	pvideo->common.pmodule = (hw_module_t*)&HAL_MODULE_INFO_SYM;
	hw_nvmedia_setvideoops(pvideo);
	/*
	* Check get pcontext.
	*/
	hw_plat_mutex_lock(&_env.mutex_context);
	if (_env.pcontext == NULL)
	{
		_env.pcontext = new HWNvmediaMultiIpcProducerAContext(pvideo);
		ret = _env.pcontext->Init();
		if (ret != 0)
		{
			HW_NVMEDIA_LOG_UNMASK("ContextPrepare fail!\r\n");
			delete(_env.pcontext);
			hw_plat_mutex_unlock(&_env.mutex_context);
			return -2;
		}
        hw_plat_event_init(&_env._main_pevent,HW_EVENT_TYPE_MANUALRESET_PROCESS_PRIVATE,0);
        _env.pcontext->SetHwEvent(&_env._main_pevent);
		_env.refcount_pcontext = 0;
	}
	pvideo->priv = _env.pcontext;
	_env.refcount_pcontext++;
	hw_plat_mutex_unlock(&_env.mutex_context);
	HW_NVMEDIA_LOG_INFO("Has init nvmedia groupa context!\r\n");
	HW_NVMEDIA_LOG_INFO("The current drive os version[%s]\r\n", HAL_MODULE_INFO_SYM.nvmedia.driveosversiondesc);
	*io_ppdevice = (struct hw_device_t*)pvideo;
    _psubthread = std::thread(hw_module_private_early_attach);
	return 0;
}

s32 hw_module_privapi_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice)
{
	struct hw_video_t* pvideo = (struct hw_video_t*)i_pdevice;
	if (HW_UNLIKELY(pvideo->common.tag != HARDWARE_DEVICE_TAG))
	{
		HW_NVMEDIA_LOG_UNMASK("unexpected video device tag[%x], should be HARDWARE_DEVICE_TAG[%x]\r\n", pvideo->common.tag, HARDWARE_DEVICE_TAG);
		return -1;
	}
	HWNvmediaMultiIpcProducerAContext* pcontext = (HWNvmediaMultiIpcProducerAContext*)pvideo->priv;
	delete(pcontext);
	delete(pvideo);
    if(_psubthread.joinable())
    {
        _mainend= true;
        _psubthread.join();
    }
	return 0;
}

s32 hw_module_private_early_attach()
{
    //wait producer ready for connect;then start early attach
    hw_plat_event_wait(&_env._main_pevent);
    _psubmodule = (struct hw_module_t *)&HAL_MODULE_INFO_SYM_ENC;
    struct hw_module_t *hmi = _psubmodule;
    struct hw_video_t *pvideo;
    void *pvoid;
    int ret =0;
    if (hw_module_privapi_init_enc(&pvoid) == 0) {
        // printf("so name[%s] init success!\r\n", i_modulesoname);
        HW_NVMEDIA_LOG_DEBUG("so details: devtype[%s]moduleid[%s]\r\n", hw_global_devtype_desc_get(hmi->devicetype), hw_video_moduleid_desc_get(hmi->module_id));
        // hmi->privdata.dso = handle;
        hmi->privdata.pvoid = pvoid;
        u32 device_api_version;
        switch (hmi->devicetype) {
            case HW_DEVICETYPE_VIDEO:
                device_api_version = HW_VIDEO_API_VERSION;
                break;
            default:
                HW_NVMEDIA_LOG_ERR("unexpected devicetype[%d]!\r\n", hmi->devicetype);
                return -1;
        }
        ret = hmi->privapi.check_device_api_version(device_api_version);
        if (ret < 0) {
            // printf("so name[%s] check_device_api_version fail!\r\n", i_modulesoname);
            return -1;
        }
        /*
         * Check tag and version one by one.
         */
        if (hmi->tag != HARDWARE_MODULE_TAG) {
            HW_NVMEDIA_LOG_ERR("check tag[%x], is not HARDWARE_MODULE_TAG[%x] fail!\r\n", hmi->tag, HARDWARE_MODULE_TAG);
            return -1;
        }
        if (hmi->hal_api_version != HARDWARE_HAL_API_VERSION) {
            HW_NVMEDIA_LOG_ERR("check hal_api_version[%x], is not HARDWARE_HAL_API_VERSION[%x] fail!\r\n", hmi->hal_api_version, HARDWARE_HAL_API_VERSION);
            return -1;
        }
        if (!HW_CHECK_MAJ_VERSION(hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION)) {
            HW_NVMEDIA_LOG_ERR("check major version is not the same, global_devicetype_version[%x], HW_GLOBAL_DEVTYPE_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
            return -1;
        }
        if (hw_global_devtype_magic_get(hmi->devicetype) != hmi->devtype_magic) {
            HW_NVMEDIA_LOG_ERR("check devtype magic fail!\r\n");
            return -1;
        }
        if (!HW_CHECK_MAJ_VERSION(hmi->device_moduleid_version, HW_DEVICE_MODULEID_VERSION)) {
            HW_NVMEDIA_LOG_ERR("check major version is not the same, device_moduleid_version[%x], HW_DEVICE_MODULEID_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
            return -1;
        }
        if (hw_video_moduleid_magic_get(hmi->module_id) != hmi->devmoduleid_magic) {
            HW_NVMEDIA_LOG_ERR("check module_id (of the specific devicetype) magic fail!\r\n");
            return -1;
        }
        HW_NVMEDIA_LOG_UNMASK("Check all of the tag and version of the module success!\r\n");
        HW_NVMEDIA_LOG_UNMASK("Finish first get the module[%s] success!\r\n", hmi->description);
    }
    ret = _psubmodule->privapi.device_get(_psubmodule, NULL, (hw_device_t **)&pvideo);
    struct hw_video_info_t videoinfo;
    struct hw_video_blockspipeline_ops_t *pblockspipeline_ops;
    // struct hw_video_blockpipeline_ops_t *pblockpipeline_ops;
    //struct hw_video_sensorpipeline_ops_t *psensorpipeline_ops;
    //struct hw_video_outputpipeline_ops_t *poutputpipeline_ops;
    hw_video_handlepipeline handlepipeline;
    //u32 blocki, sensori, numsensors, outputi;
    ret = pvideo->ops.device_open(pvideo, &videoinfo);
    if (ret < 0) {
        HW_NVMEDIA_LOG_ERR("device_open fail!\r\n");
        return -1;
    }
    HW_NVMEDIA_LOG_UNMASK("device_open success!\r\n");
    //numblocks = videoinfo.numblocks;
#ifdef BOARD_CONFIG_NVIDIA
    struct hw_video_blockspipelineconfig_t _pipelineconfig_enc =
    {
        .parrayblock = {
            [0] = {
                .bused = 1,
                .blockindex = 0,
                .parraysensor = {
                    [0] =
                    {
                        .bused = 1,
                        .blockindex = 0,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig =
                        {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 0,
                        .blockindex = 0,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        //.inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 0,
                        .blockindex = 0,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        //.inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },

                },
            },
    #if 1
            [1] = {
                .bused = 1,
                .blockindex = 1,
                .parraysensor = {
                    [0] = {
                        .bused = 1,
                        .blockindex = 1,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 0,
                        .blockindex = 1,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 0,
                        .blockindex = 1,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [3] = {
                        .bused = 0,
                        .blockindex = 1,
                        .sensorindex = 3,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                },
            },
    #endif
    #if 1
            [2] = {
                .bused = 1,
                .blockindex = 2,
                .parraysensor = {
                    [0] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [3] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 3,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .enablevic = 1,
                        .enablecuda = 1,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                },
            },
    #endif
        },
    };
#else
    struct hw_video_blockspipelineconfig_t _pipelineconfig_enc =
    {
        .parrayblock = {
            [0] = {
                .bused = 1,
                .blockindex = 0,
                .parraysensor = {
                    [0] =
                    {
                        .bused = 1,
                        .blockindex = 0,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        .datacbsconfig =
                        {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 1,
                        .blockindex = 0,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        //.inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 1,
                        .blockindex = 0,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        //.inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },

                },
            },
    #if 1
            [1] = {
                .bused = 1,
                .blockindex = 1,
                .parraysensor = {
                    [0] = {
                        .bused = 1,
                        .blockindex = 1,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 1,
                        .blockindex = 1,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 1,
                        .blockindex = 1,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [3] = {
                        .bused = 1,
                        .blockindex = 1,
                        .sensorindex = 3,
                        .bcaptureoutputrequested = 0,
                        .bisp0outputrequested = 1,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_RAW12,
                        /* .inputType = HW_VIDEO_INPUT_TYPE_YUV422, */
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                },
            },
    #endif
    #if 1
            [2] = {
                .bused = 1,
                .blockindex = 2,
                .parraysensor = {
                    [0] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 0,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [1] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 1,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [2] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 2,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                    [3] = {
                        .bused = 1,
                        .blockindex = 2,
                        .sensorindex = 3,
                        .bcaptureoutputrequested = 1,
                        .bisp0outputrequested = 0,
                        .bisp1outputrequested = 0,
                        .bisp2outputrequested = 0,
                        .enablevic = 1,
                        .enablecuda = 1,
                        .inputType = HW_VIDEO_INPUT_TYPE_YUV422,
                        .datacbsconfig = {
                            .arraynumdatacbs = 0,
                        },
                    },
                },
            },
    #endif
        },
    };
#endif
    ret = pvideo->ops.pipeline_open(pvideo, &handlepipeline, &_pipelineconfig_enc, &pblockspipeline_ops);
     if (ret < 0) {
         HW_NVMEDIA_LOG_ERR("pipeline_open fail!\r\n");
         return -1;
     }
     HW_NVMEDIA_LOG_UNMASK("pipeline_open success!\r\n");
     ret = pvideo->ops.pipeline_start(pvideo, handlepipeline);
     if (ret < 0) {
         HW_NVMEDIA_LOG_ERR("pipeline_start fail!\r\n");
         return -1;
     }

    while(!_mainend){
        //wait for main stop
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ret = pvideo->ops.pipeline_stop(pvideo, handlepipeline);
    if (ret < 0) {
        HW_NVMEDIA_LOG_ERR("pipeline_stop fail!\r\n");
        return -1;
    }
    ret = pvideo->ops.pipeline_close(pvideo, handlepipeline);
        if (ret < 0) {
            HW_NVMEDIA_LOG_ERR("pipeline_close fail!\r\n");
            return -1;
        }
        HW_NVMEDIA_LOG_UNMASK("pipeline_close success!\r\n");
        ret = pvideo->ops.device_close(pvideo);
        if (ret < 0) {
            HW_NVMEDIA_LOG_ERR("device_close fail!\r\n");
            return -1;
        }
        HW_NVMEDIA_LOG_UNMASK("device_close success!\r\n");
    _psubmodule->privapi.device_put(_psubmodule, (struct hw_device_t *)pvideo);
    return 0;
}

#endif
