#if 1
#define ENABLE_TENTORRT__ 1
#include "hw_nvmedia_groupb_impl.h"
#if(ENABLE_TENTORRT__==1)
#include <iostream>
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

extern "C" struct hw_nvmedia_groupb_module_t HAL_MODULE_INFO_SYM;

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

static u32 _testframecount_yuv422 = 0;

static u32 _binitfile_yuv422 = 0;
static FILE* _pfiletest_yuv422;
static FILE* _pfiletest_avc;
static FILE* _pfiletest_cuda;

static void handle_yuv422_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo)
{
	if (HW_UNLIKELY(_binitfile_yuv422 == 0))
	{
		//string filename = "testfile_common" + std::to_string(_blockindex) + "_" + std::to_string(_sensorindex) + fileext;
		string fileext;
		switch (i_pbufferinfo->format_maintype)
		{
		case HW_VIDEO_BUFFERFORMAT_MAINTYPE_YUV422:
			switch (i_pbufferinfo->format_subtype)
			{
			case HW_VIDEO_BUFFERFORMAT_SUBTYPE_YUV422:
				fileext = ".yuv422";
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
		_pfiletest_yuv422 = fopen(filename.c_str(), "wb");
		if (!_pfiletest_yuv422) {
			printf("Failed to create output file\r\n");
			return;
		}
		_binitfile_yuv422 = 1;
	}
	/*
	* Assume that the data is yuv422 pl(the inner pipeline already change nvidia bl format to common pl format)
	*/
	if (_testframecount_yuv422 < 3)
	{
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_yuv422);
		_testframecount_yuv422++;
	}
}

static void handle_encode_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo) {
	if (i_pbufferinfo->size < 10000) {
		return;
	}
	string fileext;
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
	string filename = "testfile" + fileext;
	remove(filename.c_str());
	_pfiletest_avc = fopen(filename.c_str(), "ab");
	if (!_pfiletest_avc) {
		printf("Failed to create output file\r\n");
		return;
	}
	fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_avc);

	
}
int ret =0;

#if 1
static void handle_cuda_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo) {
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
    preprocess_kernel_img((uint8_t *)output->data, outputWidth, outputHeight, buffer_idx, INPUT_W, INPUT_H, m_stream);
    buffer_idx += outputSize;

    gpuutils::save_rgbgpu_to_file("test111.rgb",(RGBGPUImage*)image->image,m_stream);
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
#ifdef FEATURE_OPENCV_ENABLED
    if(res.size()>0){

        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(rgbImage, res[j].bbox);
            cv::rectangle(rgbImage, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(rgbImage, std::to_string((int)res[j].class_id),
                    cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            std::cout << (int)res[j].class_id << std::endl;
            std::cout << res[j].bbox[0] << ":" << res[j].bbox[1] << ":"
                << res[j].bbox[2] << ":" << res[j].bbox[3] << ":" << std::endl;
        }
        cv::imwrite("_bus_www_test.jpg", rgbImage,params);

        delete [] phost;
    }
#endif
    /* for (size_t j = 0; j < res.size(); j++) { */
    /* std::cout << (int)res[j].class_id << std::endl; */
    /* std::cout << res[j].bbox[0] << ":" << res[j].bbox[1] << ":" */
    /* << res[j].bbox[2] << ":" << res[j].bbox[3] << ":" << std::endl; */
    /* } */
}

#else
static void handle_cuda_buffer(struct hw_video_bufferinfo_t* i_pbufferinfo) {
	//
	printf("do handle_cuda_buffer++++++++++buffer size = %d\r\n", i_pbufferinfo->size);
	if(!engineReady || i_pbufferinfo->pcustom==nullptr){
		return;
	}
	//if(ret++>0){
	//	return;
	//}
	if(1){
		//ret =1;
		string filename = "testfile.yuv";
		remove(filename.c_str());
		_pfiletest_cuda = fopen(filename.c_str(), "ab");
		if (!_pfiletest_cuda) {
			printf("Failed to create output file\r\n");
			return;
		}
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _pfiletest_cuda);
		fclose(_pfiletest_cuda);
	}
	auto h=((hw_nvmedia_cuda_datainfo_t*)i_pbufferinfo->pcustom)->height;
	auto w= ((hw_nvmedia_cuda_datainfo_t*)i_pbufferinfo->pcustom)->stride==0 ? 
		((hw_nvmedia_cuda_datainfo_t*)i_pbufferinfo->pcustom)->width :
		((hw_nvmedia_cuda_datainfo_t*)i_pbufferinfo->pcustom)->stride;
	cv::Mat yuvimg;
  	yuvimg.create(h, w, CV_8UC2);
	printf("do handle_cuda_buffer----------buffer w = %d;h=%d\r\n", w,h);
#if 1
	//memcpy(yuvimg.data, i_pbufferinfo->pbuff, h*w*3/2);
	memcpy(yuvimg.data, i_pbufferinfo->pbuff, w*h*2);
	//memcpy(yuvimg.data, i_pbufferinfo->pbuff, i_pbufferinfo->size);
	//cv::imwrite("_test.jpg", yuvimg);
	cv::Mat rgbImg;
  	cv::cvtColor(yuvimg, rgbImg, cv::COLOR_YUV2RGB_YUYV);
  	//cv::cvtColor(yuvimg, rgbImg, cv::COLOR_YUV420sp2RGB);

	float *buffer_idx = (float *)buffers[m_inputIndex];
	cv::Mat img;
  	cv::resize(rgbImg, img, cv::Size(640, 640));
  	imgs_buffer = img;
	cv::imwrite("_test.jpg", img);
  	size_t size_image = img.cols * img.rows * 3;
 	size_t size_image_dst = INPUT_H * INPUT_W * 3;
  	// copy data to pinned memory
  	memcpy(m_img_host, img.data, size_image);
	cudaMemcpyAsync(m_img_device, m_img_host, size_image,
                             cudaMemcpyHostToDevice, m_stream);	
	preprocess_kernel_img(m_img_device, img.cols, img.rows, buffer_idx, INPUT_W,
                        INPUT_H, m_stream);
  	buffer_idx += size_image_dst;
	// Run inference
  	auto start = std::chrono::system_clock::now();
  	// doInference(*m_context, m_stream, (void**)buffers, prob, 1);
  	m_context->enqueue(1, (void **)buffers, m_stream, nullptr);
	cudaMemcpyAsync(prob, (void **)buffers[1],
                             1 * OUTPUT_SIZE * sizeof(float),
                             cudaMemcpyDeviceToHost, m_stream);
	cudaStreamSynchronize(m_stream);
	auto end = std::chrono::system_clock::now();
	std::cout << "inference time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;
	std::vector<Yolo::Detection> batch_res;
  	nms(batch_res, prob, CONF_THRESH, NMS_THRESH);
  	auto &res = batch_res;
  	img = imgs_buffer;
	for (size_t j = 0; j < res.size(); j++) {
		string filename = "testfile.yuv";
    		cv::Rect r = get_rect(img, res[j].bbox);
    		cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    		//cv::putText(img, std::to_string((int)res[j].class_id),
    		cv::putText(img, filename,
                cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    		std::cout << (int)res[j].class_id << std::endl;
    		std::cout << res[j].bbox[0] << ":" << res[j].bbox[1] << ":"
              		<< res[j].bbox[2] << ":" << res[j].bbox[3] << ":" << std::endl;
  	}
	cv::imwrite("_bus.jpg", img);
#endif
}
#endif

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

	INvSIPLQueryTrace::GetInstance()->SetLevel((INvSIPLQueryTrace::TraceLevel)3);
	INvSIPLTrace::GetInstance()->SetLevel((INvSIPLTrace::TraceLevel)3);

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
	//static hw_nvmedia_cuda_context_t cuda_context;
	u32 counti;
	struct hw_video_info_t videoinfo;
    GPUImage gpuImage;
    gpuImage.out_img_type = GPU_IMG_TYPE::GPU_Bayer_RGB888;
    gpuImage.image = (void*)gpuutils::create_rgb_gpu_image(INPUT_W,INPUT_W,1,PixelLayout::NHWC_BGR,netaos::gpu::DataType::Uint8);
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
						.bcaptureoutputrequested = 1,
						.bisp0outputrequested = 0,
						.bisp1outputrequested = 0,
						.bisp2outputrequested = 0,
						.datacbsconfig =
						{
							.arraynumdatacbs = 2,
							.parraydatacbs = 
							{
								//[0] =
								//{
								//	.bused = 1,
								//	.type = HW_VIDEO_REGDATACB_TYPE_YUV422,
								//	.cb = handle_yuv422_buffer,
								//	.bsynccb = 1,
								//	// set it when you need
								//	.pcustom = nullptr,
								//},
								[0] =
								{
									.bused = 1,
									.type = HW_VIDEO_REGDATACB_TYPE_HEVC,
									.cb = handle_encode_buffer,
									.bsynccb = 1,
									// set it when you need
									.pcustom = nullptr,
								},
								[1] =
								{
									.bused = 1,
									.type = HW_VIDEO_REGDATACB_TYPE_CUDA,
									.cb = handle_cuda_buffer,
									.bsynccb = 1,
									// set it when you need
									.pcustom = &gpuImage,//&cuda_context,
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
	desEngine();
	return 0;
}

#endif

