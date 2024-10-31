#if 0

#include "hw_nvmedia_imx728_impl.h"

using namespace std;

extern "C" struct hw_nvmedia_imx728_module_t HAL_MODULE_INFO_SYM;

#define SECONDS_PER_ITERATION			1

static void thread_outputpipeline_handleframe(struct hw_video_outputpipeline_ops_t* i_poutputpipeline_ops)
{
	s32 ret;
	hw_video_handleframe handleframe;
	HW_VIDEO_FRAMERETSTATUS frameretstatus;

	uint64_t uFrameCountDelta = 0u;
	uint64_t uTimeElapsedSum = 0u;
	
	u64 prevframecount = 0;
	u64 currframecount;
	u64 timesumcurr = 0;

	auto oStartTime = chrono::steady_clock::now();
	auto timecurr = chrono::steady_clock::now();

	while (1)
	{
		ret = i_poutputpipeline_ops->gethandleframe(i_poutputpipeline_ops, &handleframe, HW_TIMEOUT_US_DEFAULT, &frameretstatus);
		if (ret == 0)
		{
			// receive notification, handle it
			if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_QUIT)
			{
				printf("Frame thread quit by restatus[quit]...\r\n");
				break;
			}
			else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_TIMEOUT)
			{
				printf("Frame receive timeout\r\n");
			}
			else if (frameretstatus == HW_VIDEO_FRAMERETSTATUS_GET)
			{
				ret = i_poutputpipeline_ops->handle(i_poutputpipeline_ops, handleframe);
				if (ret != 0)
				{
					printf("Frame handle fail!\r\n");
					break;
				}
			}
			else
			{
				printf("Sensor Unexpected notifretstatus value[%u]\r\n", frameretstatus);
			}
		}
		else
		{
			printf("Sensor Unexpected ret value[0x%x]\r\n", ret);
		}

		/*
		* Output the frame count every 2 second.
		*/
		// Wait for SECONDS_PER_ITERATION
		timecurr = chrono::steady_clock::now();
		auto uTimeElapsedMs = chrono::duration<double, std::milli>(timecurr - oStartTime).count();
		oStartTime = timecurr;
		uTimeElapsedSum += (u64)uTimeElapsedMs;
		timesumcurr += (u64)uTimeElapsedMs;
		
		if (timesumcurr > SECONDS_PER_ITERATION * 1000) {
			if (i_poutputpipeline_ops->frame_ops.getframecount(i_poutputpipeline_ops, &currframecount) != 0) {
				printf("getframecount fail!\r\n");
				break;
			}
			uFrameCountDelta = currframecount - prevframecount;
			prevframecount = currframecount;
			auto fps = (double)uFrameCountDelta / ((double)timesumcurr / 1000.0);
			string profName = "Sensor block[" + to_string(i_poutputpipeline_ops->blockindex) +
				"]sensor[" + to_string(i_poutputpipeline_ops->sensorindex) + "]_Out"
				+ to_string(i_poutputpipeline_ops->outputtype) + "\t";
			cout << profName << "Frame rate (fps):\t\t" << fps << ", frame:" << uFrameCountDelta << ", time:" << (double)((double)timesumcurr/1000.0) << "s" << endl;

			timesumcurr = 0;
		}		
	}
}

static void thread_sensorpipeline_notification(struct hw_video_sensorpipeline_ops_t* i_psensorpipeline_ops)
{
	s32 ret;
	struct hw_video_notification_t blocknotif;
	HW_VIDEO_NOTIFRETSTATUS notifretstatus;
	while (1)
	{
		ret = i_psensorpipeline_ops->getnotification(i_psensorpipeline_ops, &blocknotif, 1, HW_TIMEOUT_US_DEFAULT, &notifretstatus);
		if (ret == 0)
		{
			// receive notification, handle it
			if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_QUIT)
			{
				printf("Sensor notif thread quit by restatus[quit]...\r\n");
				break;
			}
			else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_TIMEOUT)
			{
				printf("Sensor notif receive timeout\r\n");
			}
			else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_GET)
			{
				printf("Sensor notif: %u\r\n", blocknotif.notiftype);
			}
			else
			{
				printf("Sensor Unexpected notifretstatus value[%u]\r\n", notifretstatus);
			}
		}
		else
		{
			printf("Sensor Unexpected ret value[0x%x]\r\n", ret);

		}
	}
}

static void thread_blockpipeline_notification(struct hw_video_blockpipeline_ops_t* i_pblockpipeline_ops)
{
	s32 ret;
	struct hw_video_notification_t blocknotif;
	HW_VIDEO_NOTIFRETSTATUS notifretstatus;
	while (1)
	{
		ret = i_pblockpipeline_ops->getnotification(i_pblockpipeline_ops, &blocknotif, 1, HW_TIMEOUT_US_DEFAULT, &notifretstatus);
		if (ret == 0)
		{
			// receive notification, handle it
			if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_QUIT)
			{
				printf("Block notif thread quit by restatus[quit]...\r\n");
				break;
			}
			else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_TIMEOUT)
			{
				printf("Block notif receive timeout\r\n");
			}
			else if (notifretstatus == HW_VIDEO_NOTIFRETSTATUS_GET)
			{
				printf("Block notif: %u\r\n", blocknotif.notiftype);
			}
			else
			{
				printf("Block Unexpected notifretstatus value[%u]\r\n", notifretstatus);
			}
		}
		else
		{
			printf("Block Unexpected ret value[0x%x]\r\n", ret);
			
		}
	}
}

static u32 _testframecount = 0;

static u32 _binitfile = 0;
static FILE* _pfiletest;

static void handle_yuv420_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
	if (HW_UNLIKELY(_binitfile == 0))
	{
		//string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
		string fileext;
		switch (i_pbufferinfo->format_maintype)
		{
		case HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV422:
			fileext = ".yuv422";
			break;
		default:
			printf("Unexpected type!\r\n");
			break;
		}
		string filename = "testfile_type_" + std::to_string(i_pbufferinfo->format_subtype) + fileext;
		remove(filename.c_str());
		_pfiletest = fopen(filename.c_str(), "wb");
		if (!_pfiletest) {
			printf("Failed to create output file\r\n");
			return;
		}
		_binitfile = 1;
	}
	/*
	* Assume that the data is yuv420 pl(the inner pipeline already change nvidia bl format to common pl format)
	*/
	if (_testframecount < 3)
	{
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest);
		_testframecount++;
	}
}

int main()
{
    /*
    * Register default sig handler.
    */
    hw_plat_regsighandler_default();

    struct hw_module_t* pmodule = (struct hw_module_t*)&HAL_MODULE_INFO_SYM;
	struct hw_module_t* hmi = pmodule;
	struct hw_video_t* pvideo;
    void* pvoid;
	s32 ret;
    if (hw_module_privapi_init(&pvoid) == 0)
    {
		//printf("so name[%s] init success!\r\n", i_modulesoname);
		printf("so details: devtype[%s]moduleid[%s]\r\n", hw_global_devtype_desc_get(hmi->devicetype), hw_video_moduleid_desc_get(hmi->module_id));
        //hmi->privdata.dso = handle;
        hmi->privdata.pvoid = pvoid;
        u32 device_api_version;
		switch (hmi->devicetype)
		{
		case HW_DEVICETYPE_VIDEO:
			device_api_version = HW_VIDEO_API_VERSION;
			break;
		default:
			printf("unexpected devicetype[%d]!\r\n", hmi->devicetype);
			return -1;
		}
		ret = hmi->privapi.check_device_api_version(device_api_version);
		if (ret < 0)
		{
			//printf("so name[%s] check_device_api_version fail!\r\n", i_modulesoname);
			return -1;
		}
		/*
		* Check tag and version one by one.
		*/
		if (hmi->tag != HARDWARE_MODULE_TAG)
		{
			printf("check tag[%x], is not HARDWARE_MODULE_TAG[%x] fail!\r\n", hmi->tag, HARDWARE_MODULE_TAG);
			return -1;
		}
		if (hmi->hal_api_version != HARDWARE_HAL_API_VERSION)
		{
			printf("check hal_api_version[%x], is not HARDWARE_HAL_API_VERSION[%x] fail!\r\n", hmi->hal_api_version, HARDWARE_HAL_API_VERSION);
			return -1;
		}
		if (!HW_CHECK_MAJ_VERSION(hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION))
		{
			printf("check major version is not the same, global_devicetype_version[%x], HW_GLOBAL_DEVTYPE_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
			return -1;
		}
		if (hw_global_devtype_magic_get(hmi->devicetype) != hmi->devtype_magic)
		{
			printf("check devtype magic fail!\r\n");
			return -1;
		}
		if (!HW_CHECK_MAJ_VERSION(hmi->device_moduleid_version, HW_DEVICE_MODULEID_VERSION))
		{
			printf("check major version is not the same, device_moduleid_version[%x], HW_DEVICE_MODULEID_VERSION[%x]!\r\n", hmi->global_devicetype_version, HW_GLOBAL_DEVTYPE_VERSION);
			return -1;
		}
		if (hw_video_moduleid_magic_get(hmi->module_id) != hmi->devmoduleid_magic)
		{
			printf("check module_id (of the specific devicetype) magic fail!\r\n");
			return -1;
		}
		printf("Check all of the tag and version of the module success!\r\n");
		printf("Finish first get the module[%s] success!\r\n", hmi->description);
    }
	ret = pmodule->privapi.device_get(pmodule, NULL, (hw_device_t**)&pvideo);
	u32 counti;
	struct hw_video_info_t videoinfo;
	struct hw_video_blockspipelineconfig_t pipelineconfig =
	{
		.parrayblock =
		{
			[0] = 
			{
				.bused = 1,
				.blockindex = 0,
				.parraysensor = 
				{
					[0] = 
					{
						.bused = 1,
						.blockindex = 0,
						.sensorindex = 0,
						.bcaptureoutputrequested = 0,
						.bisp0outputrequested = 1,
						.bisp1outputrequested = 0,
						.bisp2outputrequested = 0,
						.datacbsconfig =
						{
							.arraynumdatacbs = 1,
							.parraydatacbs = 
							{
								[0] =
								{
									.bused = 1,
									.type = HW_VIDEO_REGDATACB_TYPE_YUV420,
									.cb = handle_yuv420_buffer,
									.bsynccb = 1,
								},
							},
						},
					},
				},
			},
		},
	};
	struct hw_video_blockspipeline_ops_t* pblockspipeline_ops;
	struct hw_video_blockpipeline_ops_t* pblockpipeline_ops;
	struct hw_video_sensorpipeline_ops_t* psensorpipeline_ops;
	struct hw_video_outputpipeline_ops_t* poutputpipeline_ops;
	hw_video_handlepipeline handlepipeline;
	u32 blocki, numblocks, sensori, numsensors, outputi;
	std::vector<std::unique_ptr<std::thread>> vthreadpipelinenotif, vthreadblocknotif, vthreadoutput;
	for (counti = 0; counti < 3; counti++)
	{
		ret = pvideo->ops.device_open(pvideo, &videoinfo);
		if (ret < 0) {
			printf("device_open fail!\r\n");
			return ret;
		}
		printf("device_open success!\r\n");
		numblocks = videoinfo.numblocks;
		ret = pvideo->ops.pipeline_open(pvideo, &handlepipeline, &pipelineconfig, &pblockspipeline_ops);
		if (ret < 0) {
			printf("pipeline_open fail!\r\n");
			return ret;
		}
		printf("pipeline_open success!\r\n");
#if 1
		for (blocki = 0; blocki < numblocks; blocki++)
		{
			pblockpipeline_ops = &pblockspipeline_ops->parrayblock[blocki];
			if (pblockpipeline_ops->bused)
			{
				vthreadblocknotif.push_back(std::make_unique<std::thread>(thread_blockpipeline_notification,
					pblockpipeline_ops));
				numsensors = pblockpipeline_ops->numsensors;
				for (sensori = 0; sensori < numsensors; sensori++)
				{
					psensorpipeline_ops = &pblockpipeline_ops->parraysensor[sensori];
					if (psensorpipeline_ops->bused)
					{
						vthreadpipelinenotif.push_back(std::make_unique<std::thread>(thread_sensorpipeline_notification,
							psensorpipeline_ops));
						for (outputi = 0; outputi <= HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAX; outputi++)
						{
							poutputpipeline_ops = &psensorpipeline_ops->parrayoutput[outputi];
							if (poutputpipeline_ops->bused)
							{
								vthreadoutput.push_back(std::make_unique<std::thread>(thread_outputpipeline_handleframe,
									poutputpipeline_ops));
							}
						}
					}
				}
			}
		}
		ret = pvideo->ops.pipeline_start(pvideo, handlepipeline);
		if (ret < 0) {
			printf("pipeline_start fail!\r\n");
			return ret;
		}
		printf("sleep 3 seconds.\r\n");
		sleep(3);
		ret = pvideo->ops.pipeline_stop(pvideo, handlepipeline);
		if (ret < 0) {
			printf("pipeline_stop fail!\r\n");
			return ret;
		}
		printf("pipeline_stop success!\r\n");
		for (auto& upthread : vthreadpipelinenotif) {
			if (upthread != nullptr) {
				upthread->join();
				upthread.reset();
			}
		}
		for (auto& upthread : vthreadblocknotif) {
			if (upthread != nullptr) {
				upthread->join();
				upthread.reset();
			}
		}
		for (auto& upthread : vthreadoutput) {
			if (upthread != nullptr) {
				upthread->join();
				upthread.reset();
			}
		}
#endif
		ret = pvideo->ops.pipeline_close(pvideo, handlepipeline);
		if (ret < 0) {
			printf("pipeline_close fail!\r\n");
			return ret;
		}
		printf("pipeline_close success!\r\n");
		ret = pvideo->ops.device_close(pvideo);
		if (ret < 0) {
			printf("device_close fail!\r\n");
			return ret;
		}
		printf("device_close success!\r\n");
	}
	ret = pmodule->privapi.device_put(pmodule, (struct hw_device_t*)pvideo);
	if (ret < 0) {
		printf("device_put fail!\r\n");
		return ret;
	}
	printf("device_put success!\r\n");
	return 0;
}

#endif
