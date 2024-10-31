#if 1

#define ENABLE_TENTORRT__ 0
#include "hw_nvmedia_groupa_impl.h"
#if(ENABLE_TENTORRT__==1)
#include <iostream>
#include <fstream>
/* #include <opencv2/opencv.hpp> */
#include "logging.hpp"
#include "yololayer.h"
#include "preprocess.h"
#include "common.hpp"
#include "cuda_utils.h"

#include "hw_nvmedia_gpu_common.hpp"
#include "gpu_convert.hpp"
static Logger gLogger;
// ensure it exceed the maximum size in the input images !
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define CONF_THRESH 0.5
#define NMS_THRESH 0.4


static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE =
Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;


static float prob[OUTPUT_SIZE];
float* buffers[2];
nvinfer1::IRuntime* m_runtime;
nvinfer1::ICudaEngine* m_engine;
nvinfer1::IExecutionContext* m_context;
cudaStream_t m_stream;
int m_inputIndex;
int m_outputIndex;
uint8_t* m_img_host = nullptr;
uint8_t* m_img_device = nullptr;
bool engineReady = false;
/* cv::Mat imgs_buffer; */
#endif
using namespace std;

extern "C" struct hw_nvmedia_groupa_module_t HAL_MODULE_INFO_SYM;

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

static u32 _testframecount_yuv420 = 0;

static u32 _binitfile_yuv420 = 0;
static FILE* _pfiletest_yuv420;
static FILE* _pfiletest_cuda;

static void handle_yuv420_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
	if (HW_UNLIKELY(_binitfile_yuv420 == 0))
	{
		//string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
		string fileext;
		switch (i_pbufferinfo->format_maintype)
		{
		case HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV420:
			switch (i_pbufferinfo->format_subtype)
			{
			case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420:
				fileext = ".yuv420pl";
				break;
			case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV420_PRIV:
				fileext = ".yuv420bl";
				break;
			default:
				printf("Unexpected sub type!\r\n");
				return;
			}
			break;
		default:
			printf("Unexpected main type!\r\n");
			return;
		}
		string filename = "testfile" + fileext;
		remove(filename.c_str());
		_pfiletest_yuv420 = fopen(filename.c_str(), "wb");
		if (!_pfiletest_yuv420) {
			printf("Failed to create output file\r\n");
			return;
		}
		_binitfile_yuv420 = 1;
	}
	/*
	* Assume that the data is yuv420 pl(the inner pipeline already change nvidia bl format to common pl format)
	*/
	if (_testframecount_yuv420 < 3)
	{
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_yuv420);
		_testframecount_yuv420++;
	}
}


static u32 _testframecount_rgb = 0;

static u32 _binitfile_rgb = 0;
static FILE* _pfiletest_rgb;

static void handle_raw12_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
	if (HW_UNLIKELY(_binitfile_rgb == 0))
	{
		//string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
		string fileext;
		switch (i_pbufferinfo->format_maintype)
		{
		case HW_VIDEO_BUFFERFORMAT_SUBTYPE_RAW12:
			fileext = ".raw12";
			break;
		default:
			printf("Unexpected main type!\r\n");
			return;
		}
		string filename = "testfile" + fileext;
		remove(filename.c_str());
		_pfiletest_rgb = fopen(filename.c_str(), "wb");
		if (!_pfiletest_rgb) {
			printf("Failed to create output file\r\n");
			return;
		}
		_binitfile_rgb = 1;
	}
	/*
	* Assume that the data is rgb pl(the inner pipeline already change nvidia bl format to common pl format)
	*/
	if (_testframecount_rgb < 3)
	{
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_rgb);
		_testframecount_rgb++;
	}
}
#if 0
static void handle_cuda_buffer_origin(struct hw_video_bufferinfo_t* i_pbufferinfo) {
    printf("do handle_cuda_buffer++++++++++buffer size = %d\r\n", i_pbufferinfo->size);
    if(!engineReady || i_pbufferinfo->pcustom==nullptr){
        return;
    }
    size_t outputWidth = INPUT_H;
     size_t outputHeight = INPUT_W;
     size_t outputSize = 0;
     outputSize = outputWidth * outputHeight * 3;
    GPUImage *image = (GPUImage*)i_pbufferinfo->pcustom;
    float *buffer_idx = (float *)buffers[m_inputIndex];
    RGBGPUImage* output = (RGBGPUImage*)image->image;
    preprocess_kernel_img((uint8_t *)output->data, (int)outputWidth, (int)outputHeight, buffer_idx, INPUT_W, INPUT_H, m_stream);
    buffer_idx += outputSize;

    /* gpuutils::save_rgbgpu_to_file("test.rgb",(RGBGPUImage*)image->image,m_stream); */
    // Run inference
    auto start = std::chrono::system_clock::now();
    // doInference(*m_context, m_stream, (void**)buffers, prob, 1);
    m_context->enqueue(1, (void **)buffers, m_stream, nullptr);
    cudaMemcpyAsync(prob, (void **)buffers[1],
            1 * OUTPUT_SIZE * sizeof(float),
            cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    
#ifdef FEATURE_OPENCV_ENABLED
    //opencv dump
    size_t num_element    =outputSize;
    uint8_t* phost   = new uint8_t[num_element];
    cudaMemcpyAsync(phost, output->data, num_element, ::cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    cv::Mat rgbImage(640, 640, CV_8UC3, phost);
    // 现在可以将 Mat 对象保存为 JPEG 文件。
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
    params.push_back(90);  // JPEG 压缩质量为 90。
    cv::imwrite("_bus_www_test.jpg", rgbImage,params);

    delete [] phost;
#endif


    /* cv::Mat rgbImg; */
    /* rgbImg.create(outputWidth,outputHeight,CV_8UC3); */
    /* memcpy(rgbImg.data, (uint8_t*)output->data, outputWidth*outputHeight*3); */
    /* {//dump */
    /*         std::fstream fout("proc_out.rgb", ios::binary | ios::out); */
    /*         if(fout.good()){ */
    /*             /1* std::fprintf(stderr, "Can not open %s\n", file.c_str()); *1/ */
    /*             fout.write((char*)buffers[m_inputIndex], outputSize); */
    /*         } */

    /*     } */
    auto end = std::chrono::system_clock::now();
    std::cout << "inference time: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                start)
        .count()
        << "ms" << std::endl;
    std::vector<Yolo::Detection> batch_res;
    nms(batch_res, prob, CONF_THRESH, NMS_THRESH);
    auto &res = batch_res;
    std::cout<< "res.size="<<res.size()<<std::endl;
    /* for (size_t j = 0; j < res.size(); j++) { */
    /* std::cout << (int)res[j].class_id << std::endl; */
    /* std::cout << res[j].bbox[0] << ":" << res[j].bbox[1] << ":" */
    /* << res[j].bbox[2] << ":" << res[j].bbox[3] << ":" << std::endl; */
    /* } */
}
#endif

static u32 _btestrgbonce = 0;
static FILE* _pfiletest_avc;
static int flag =0;

static void handle_encode_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo) {
	printf("do handle_encode_buffer++++++++++buffer+ bufsize = %d\r\n",i_pbufferinfo->size);
	if (i_pbufferinfo->size < 10000||flag) {
		return;
	}
	string fileext;
	fileext = ".h264";
#if 0
	switch (i_pbufferinfo->format_maintype)
	{
	case HW_VIDEO_BUFFERFORMAT_MAINTYPE_AVC:
		switch (i_pbufferinfo->format_subtype)
		{
		case HW_VIDEO_BUFFERFORMAT_SUBTYPE_AVC:
			fileext = ".h264";
			break;
		default:
			printf("Unexpected sub type!\r\n");
			return;
		}
		break;
	default:
		printf("Unexpected main type!\r\n");
		return;
	}
#endif
	string filename = "testfile" + fileext;
	remove(filename.c_str());
	_pfiletest_avc = fopen(filename.c_str(), "ab+");
	if (!_pfiletest_avc) {
		printf("Failed to create output file\r\n");
		return;
	}
	ssize_t res = fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_avc);
	fclose(_pfiletest_avc);
	flag = 1;
	printf("write buffszie=%d\n",i_pbufferinfo->size);
}

static void handle_cuda_buffer(struct hw_video_cudabufferinfo_t* i_pbufferinfo) {
	printf("do handle_cuda_buffer++++++++++buffer\r\n");
#if 0
	if (!engineReady) {
		return;
	}
	size_t outputWidth = INPUT_H;
	size_t outputHeight = INPUT_W;
	size_t outputSize = 0;
	outputSize = outputWidth * outputHeight * 3;
	float* buffer_idx = (float*)buffers[m_inputIndex];
	preprocess_kernel_img((uint8_t*)i_pbufferinfo->gpuinfo.rgbinfo.pbuff, outputWidth, outputHeight, buffer_idx, INPUT_W, INPUT_H, m_stream);
	buffer_idx += outputSize;

	if (_btestrgbonce == 0) {
		u32 buffsize = i_pbufferinfo->gpuinfo.rgbinfo.buffsize;
		u8* phost = new u8[buffsize];
		cudaMemcpyAsync(phost, i_pbufferinfo->gpuinfo.rgbinfo.pbuff, buffsize, ::cudaMemcpyDeviceToHost, m_stream);
		cudaStreamSynchronize(m_stream);
		std::fstream fout("test.rgb", std::ios::binary | std::ios::out);
		fout.write((char*)phost, buffsize);
		//gpuutils::save_rgbgpu_to_file("test.rgb", (RGBGPUImage*)image->image, m_stream);
		_btestrgbonce = 1;
	}
	// Run inference
	auto start = std::chrono::system_clock::now();
	// doInference(*m_context, m_stream, (void**)buffers, prob, 1);
	m_context->enqueue(1, (void**)buffers, m_stream, nullptr);
	cudaMemcpyAsync(prob, (void**)buffers[1],
		1 * OUTPUT_SIZE * sizeof(float),
		cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);

#ifdef FEATURE_OPENCV_ENABLED
	//opencv dump
	size_t num_element = outputSize;
	uint8_t* phost = new uint8_t[num_element];
	cudaMemcpyAsync(phost, i_pbufferinfo->gpuinfo.rgbinfo.pbuff, num_element, ::cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	cv::Mat rgbImage(640, 640, CV_8UC3, phost);
	// 现在可以将 Mat 对象保存为 JPEG 文件。
	std::vector<int> params;
	params.push_back(cv::IMWRITE_JPEG_QUALITY);
	params.push_back(90);  // JPEG 压缩质量为 90。
	cv::imwrite("_bus_www_test.jpg", rgbImage, params);

	delete[] phost;
#endif


	/* cv::Mat rgbImg; */
	/* rgbImg.create(outputWidth,outputHeight,CV_8UC3); */
	/* memcpy(rgbImg.data, (uint8_t*)output->data, outputWidth*outputHeight*3); */
	/* {//dump */
	/*         std::fstream fout("proc_out.rgb", ios::binary | ios::out); */
	/*         if(fout.good()){ */
	/*             /1* std::fprintf(stderr, "Can not open %s\n", file.c_str()); *1/ */
	/*             fout.write((char*)buffers[m_inputIndex], outputSize); */
	/*         } */

	/*     } */
	auto end = std::chrono::system_clock::now();
	std::cout << "inference time: "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end -
			start)
		.count()
		<< "ms" << std::endl;
	std::vector<Yolo::Detection> batch_res;
	nms(batch_res, prob, CONF_THRESH, NMS_THRESH);
	auto& res = batch_res;
	std::cout << "res.size=" << res.size() << std::endl;
	/* for (size_t j = 0; j < res.size(); j++) { */
	/* std::cout << (int)res[j].class_id << std::endl; */
	/* std::cout << res[j].bbox[0] << ":" << res[j].bbox[1] << ":" */
	/* << res[j].bbox[2] << ":" << res[j].bbox[3] << ":" << std::endl; */
	/* } */
#endif
}

static void createEngine() {
#if( ENABLE_TENTORRT__ == 1)
	/*------------------------------load yolov5.engine--------------------------------------*/
	std::string engine_name = "yolov5.engine";
	// deserialize the .engine and run inference
	std::ifstream file(engine_name, std::ios::binary);
	if (!file.good()) {
		printf("read %s error\r\n", engine_name.c_str());
	}
	char* trtModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	file.read(trtModelStream, size);
	printf("read engine ----size=%d\r\n", size);
	file.close();

	m_runtime = nvinfer1::createInferRuntime(gLogger);
	if (m_runtime == nullptr) {
		printf("m_runtime == nullptr\r\n");
		return;
	}
	m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
	if (m_engine == nullptr) {
		printf("m_engine == nullptr\r\n");
		return;
	}
	m_context = m_engine->createExecutionContext();
	if (m_context == nullptr) {
		printf("m_context == nullptr\r\n");
		return;
	}
	delete[] trtModelStream;
	if (m_engine->getNbBindings() != 2) {
		return;
	}

	m_inputIndex = m_engine->getBindingIndex("data");
	m_outputIndex = m_engine->getBindingIndex("prob");

	cudaMalloc((void**)&buffers[m_inputIndex],
		3 * INPUT_H * INPUT_W * sizeof(float));
	cudaMalloc((void**)&buffers[m_outputIndex], OUTPUT_SIZE * sizeof(float));

	cudaError_t err = cudaStreamCreate(&m_stream);
	err = cudaMallocHost((void**)&m_img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3);
	err = cudaMalloc((void**)&m_img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3);
	engineReady = true;
#endif
}
static void desEngine() {
#if( ENABLE_TENTORRT__ == 1)
	cudaStreamDestroy(m_stream);
	CUDA_CHECK(cudaFree(m_img_device));
	CUDA_CHECK(cudaFreeHost(m_img_host));
	CUDA_CHECK(cudaFree(buffers[m_inputIndex]));
	CUDA_CHECK(cudaFree(buffers[m_outputIndex]));


	// Destroy the engine
	m_context->destroy();
	m_engine->destroy();
	m_runtime->destroy();
#endif
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
	createEngine();
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
    //GPUImage gpuImage;
    //gpuImage.out_img_type = GPU_IMG_TYPE::GPU_Bayer_RGB888;
    //gpuImage.image = (void*)gpuutils::create_rgb_gpu_image(INPUT_W,INPUT_W,1,PixelLayout::NHWC_BGR,netaos::gpu::DataType::Uint8);
    /* gpuImage.image = (void*)gpuutils::create_rgb_gpu_image(INPUT_W,INPUT_W,1,PixelLayout::NCHW_RGB,netaos::gpu::DataType::Uint8,buffers[m_inputIndex]); */
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
						.bisp1outputrequested = 1,
						.bisp2outputrequested = 0,
						.enablevic=0,
						.enableenc=1,
						.enablecommon=1,
						.enablecuda=1,
						.inputType=HW_VIDEO_INPUT_TYPE_RAW12,
						.datacbsconfig =
						{
							.arraynumdatacbs = 2,
							.parraydatacbs =
							{
								[0] =
								{
									.bused = 1,
									.type = HW_VIDEO_REGDATACB_TYPE_CUDA,
									.busecaptureresolution = 0,
									.customwidth = 640,
									.customheight = 640,
									.busecaptureframerate = 1,
									.rotatedegrees = 0,
									//.cb = handle_cuda_buffer,
									.cudaconfig = 
									{
										.imgtype = HW_VIDEO_REGCUDADATACB_IMGTYPE_RGB888_NHWC_BGR,
										.interpolation = HW_VIDEO_REGCUDADATACB_INTERPOLATION_NEAREST,
									},
									.cudacb = handle_cuda_buffer,
									.bsynccb = 1,
									// set it when you need
									.pcustom = nullptr,
									//.pcustom = &gpuImage,//&cuda_context,
								},
								[1] =
								{
									.bused = 1,
									//.type = HW_VIDEO_REGDATACB_TYPE_YUV420_PRIV,
									.type = HW_VIDEO_REGDATACB_TYPE_HEVC,
									.busecaptureresolution = 1,
									.busecaptureframerate = 1,
									.rotatedegrees = 0,
									.cb = handle_encode_buffer,
									.bsynccb = 1,
									// set it when you need
									.pcustom = nullptr,
								},
								//[1] =
								//{
								//	.bused = 1,
								//	.type = HW_VIDEO_REGDATACB_TYPE_YUV420,
								//	.busecaptureresolution = 1,
								//	.busecaptureframerate = 1,
								//	.rotatedegrees = 0,
								//	.cb = handle_yuv420_buffer,
								//	.bsynccb = 1,
								//	// set it when you need
								//	.pcustom = nullptr,
								//},
								/* [1] = */
								/* { */
								/* 	.bused = 0, */
								/* 	.type = HW_VIDEO_REGDATACB_TYPE_RAW12, */
								/* 	.cb = handle_raw12_buffer, */
								/* 	.bsynccb = 1, */
								/* 	// set it when you need */
								/* 	.pcustom = nullptr, */
								/* }, */
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
	desEngine();
	return 0;
}

#endif
