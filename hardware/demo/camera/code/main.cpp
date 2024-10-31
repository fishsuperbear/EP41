#include "hal_camera.hpp"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include <unistd.h>

using namespace std;

unique_ptr<ICameraDevice> _pcameradevice;

CAMERA_DEVICE_OPENTYPE _opentype = CAMERA_DEVICE_OPENTYPE_GROUPA_SENSOR_ONE_IMX728;
//CAMERA_DEVICE_OPENTYPE _opentype = CAMERA_DEVICE_OPENTYPE_GROUPB_SENSOR_ONE_ISX021;
//CAMERA_DEVICE_OPENTYPE _opentype = CAMERA_DEVICE_OPENTYPE_GROUPC_SENSOR_FOUR_OVXIF;

/*
* 0 or 1, when 1 mean using origin capture width and height etc..
* when 0 mean using custom width and height which you set to CameraDeviceDataCbRegInfo.
*/
#define USING_ORIGIN_CAPTURE					1

static u32 _testframecount_yuv420 = 0;

static u32 _binitfile_yuv420 = 0;
static FILE* _pfiletest_yuv420;

void handle_yuv420_buffer(CameraDeviceDataCbInfo* i_pbufferinfo)
{
    if (_binitfile_yuv420 == 0)
    {
        string filename = "testfile.yuv420pl";
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


static u32 _testframecount_yuv422 = 0;

static u32 _binitfile_yuv422 = 0;
static FILE* _pfiletest_yuv422;

void handle_yuv422_buffer(CameraDeviceDataCbInfo* i_pbufferinfo)
{
	if (_binitfile_yuv422 == 0)
	{
		string filename = "testfile.yuv422pl";
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

static u32 _parray_testframecount_index_yuv422[4] = { 0 };

static u32 _binitfile_index_yuv422 = 0;

static FILE* _parray_pfiletest_index_yuv422[4] = { nullptr };

void handle_yuv422_buffer_index(CameraDeviceDataCbInfo* i_pbufferinfo)
{
	/*
	* Assume that the data is yuv422 pl(the inner pipeline already change nvidia bl format to common pl format)
	*/
	if (_parray_testframecount_index_yuv422[i_pbufferinfo->sensorindex] < 3)
	{
		fwrite(i_pbufferinfo->pbuff, i_pbufferinfo->size, 1U, _parray_pfiletest_index_yuv422[i_pbufferinfo->sensorindex]);
		_parray_testframecount_index_yuv422[i_pbufferinfo->sensorindex]++;
	}
}

static u32 _btestrgbonce = 0;
static cudaStream_t _cudastream;

static void handle_cuda_buffer(struct CameraDeviceGpuDataCbInfo* i_pbufferinfo) 
{
	printf("do handle_cuda_buffer++++++++++buffer\r\n");

	if (_btestrgbonce == 0) {
		u32 buffsize = i_pbufferinfo->pgpuinfo->prgbinfo->buffsize;
		u8* phost = new u8[buffsize];
		cudaError_t err = cudaStreamCreate(&_cudastream);
		cudaMemcpyAsync(phost, i_pbufferinfo->pgpuinfo->prgbinfo->pbuff, buffsize, ::cudaMemcpyDeviceToHost, _cudastream);
		cudaStreamSynchronize(_cudastream);
		std::fstream fout("test.rgb", std::ios::binary | std::ios::out);
		fout.write((char*)phost, buffsize);
		//gpuutils::save_rgbgpu_to_file("test.rgb", (RGBGPUImage*)image->image, m_stream);
		_btestrgbonce = 1;
	}
}


void handle_event(CameraDeviceEventCbInfo* i_peventcbinfo)
{
	// currently do nothing
}


class CameraDeviceCallbackOrin : public ICameraDeviceCallback
{
public:
	virtual CAMERA_DEVICE_OPENTYPE RegisterOpenType()
	{
		return _opentype;
	}
	virtual CAMERA_DEVICE_OPENMODE RegisterOpenMode()
	{
		return CAMERA_DEVICE_OPENMODE_MAIN;
	}
	virtual s32 RegisterCallback()
	{
		s32 ret = 0;
		/*
		* Register event callback.
		*/
		CameraDeviceEventCbRegInfo eventreginfo;
		eventreginfo.opentype = _opentype;
		// see blocktype note of CameraDeviceDataCbRegInfo
		eventreginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_CURRENT_SINGLE_GROUP;
		// currently only one sensor type in one block, need to change it in the future
		eventreginfo.sensortype = CAMERA_DEVICE_SENSOR_TYPE_CURRENT_SINGLE_SENSOR;
		eventreginfo.sensorindex = 0;
		if ((ret = _pcameradevice->RegisterEventCallback(&eventreginfo, handle_event)) < 0) {
			printf("RegisterEventCallback fail! ret=%x\r\n", ret);
		}
		/*
		* Register data callback and gpu data callback.
		*/
		CameraDeviceDataCbRegInfo datareginfo, reginfo_yuv420;
		CameraDeviceGpuDataCbRegInfo gpudatareginfo;
		datareginfo.opentype = _opentype;
		// see blocktype note of CameraDeviceDataCbRegInfo
		datareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_CURRENT_SINGLE_GROUP;
		// currently only one sensor type in one block, need to change it in the future
		datareginfo.sensortype = CAMERA_DEVICE_SENSOR_TYPE_CURRENT_SINGLE_SENSOR;
		datareginfo.sensorindex = 0;
		if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPA_SENSOR_ONE_IMX728) {
			datareginfo.datacbtype = CAMERA_DEVICE_DATACB_TYPE_YUV420;
		}
		else if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPB_SENSOR_ONE_ISX021) {
			datareginfo.datacbtype = CAMERA_DEVICE_DATACB_TYPE_YUV422;
		}
		else if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPC_SENSOR_FOUR_OVXIF) {
			datareginfo.datacbtype = CAMERA_DEVICE_DATACB_TYPE_YUV422;
		}

#if (USING_ORIGIN_CAPTURE == 1)
		datareginfo.busecaptureresolution = 1;
#else
		datareginfo.busecaptureresolution = 0;
		// imx728:3840*2160, isx021:1920*1080, ovx1f:1280*960
		datareginfo.customwidth = 3840;
		datareginfo.customheight = 2160;
#endif
		datareginfo.busecaptureframerate = 1;
		// set it to 0 currently
		datareginfo.rotatedegrees = 0;
		// you can set your custom context pointer
		datareginfo.pcontext = nullptr;

		/*
		* Gpu data register information.
		*/
		gpudatareginfo.opentype = _opentype;
		// see blocktype note of CameraDeviceDataCbRegInfo
		gpudatareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_CURRENT_SINGLE_GROUP;
		// currently only one sensor type in one block, need to change it in the future
		gpudatareginfo.sensortype = CAMERA_DEVICE_SENSOR_TYPE_CURRENT_SINGLE_SENSOR;
		gpudatareginfo.sensorindex = 0;
		gpudatareginfo.gpuimgtype = CAMERA_DEVICE_GPUDATACB_IMGTYPE_RGB888_NHWC_BGR;
		gpudatareginfo.interpolation = CAMERA_DEVICE_GPUDATACB_INTERPOLATION_NEAREST;
		gpudatareginfo.busecaptureresolution = 0;
		gpudatareginfo.customwidth = 640;
		gpudatareginfo.customheight = 640;
		gpudatareginfo.busecaptureframerate = 1;
		gpudatareginfo.rotatedegrees = 0;
		gpudatareginfo.pcontext = nullptr;

		if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPA_SENSOR_ONE_IMX728) {
			/*
			* Data callback.
			*/
			if ((ret = _pcameradevice->RegisterDataCallback(&datareginfo, handle_yuv420_buffer)) < 0) {
				printf("RegisterDataCallback fail! ret=%x\r\n", ret);
			}
			/*
			* Gpu data callback.
			*/
			if ((ret = _pcameradevice->RegisterGpuDataCallback(&gpudatareginfo, handle_cuda_buffer)) < 0) {
				printf("RegisterDataCallback fail! ret=%x\r\n", ret);
			}
		}
		else if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPB_SENSOR_ONE_ISX021) {
			if ((ret = _pcameradevice->RegisterDataCallback(&datareginfo, handle_yuv422_buffer)) < 0) {
				printf("RegisterDataCallback handle_yuv422_buffer fail! ret=%x\r\n", ret);
			}
			reginfo_yuv420 = datareginfo;
			reginfo_yuv420.datacbtype = CAMERA_DEVICE_DATACB_TYPE_YUV420;
			if ((ret = _pcameradevice->RegisterDataCallback(&reginfo_yuv420, handle_yuv420_buffer)) < 0) {
				printf("RegisterDataCallback handle_yuv420_buffer fail! ret=%x\r\n", ret);
			}
		}
		else if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPC_SENSOR_FOUR_OVXIF) {
			if ((ret = _pcameradevice->RegisterDataCallback(&datareginfo, handle_yuv422_buffer_index)) < 0) {
				printf("RegisterDataCallback fail! ret=%x\r\n", ret);
			}
		}
		if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPC_SENSOR_FOUR_OVXIF) {
			u32 sensorindex;
			for (sensorindex = 1; sensorindex < 4; sensorindex++)
			{
				datareginfo.opentype = _opentype;
				// see blocktype note of CameraDeviceDataCbRegInfo
				datareginfo.blocktype = CAMERA_DEVICE_BLOCK_TYPE_CURRENT_SINGLE_GROUP;
				// currently only one sensor type in one block, need to change it in the future
				datareginfo.sensortype = CAMERA_DEVICE_SENSOR_TYPE_CURRENT_SINGLE_SENSOR;
				datareginfo.sensorindex = sensorindex;
				datareginfo.datacbtype = CAMERA_DEVICE_DATACB_TYPE_YUV422;
#if (USING_ORIGIN_CAPTURE == 1)
				datareginfo.busecaptureresolution = 1;
#else
				datareginfo.busecaptureresolution = 1;
				// imx728:3840*2160, isx021:1920*1080, ovx1f:1280*960
				datareginfo.customwidth = 3840;
				datareginfo.customheight = 2160;
#endif
				// set it to 0 currently
				datareginfo.rotatedegrees = 0;
				// you can set your custom context pointer
				datareginfo.pcontext = nullptr;
				if ((ret = _pcameradevice->RegisterDataCallback(&datareginfo, handle_yuv422_buffer_index)) < 0) {
					printf("RegisterDataCallback fail! ret=%x\r\n", ret);
				}
			}
			
		}
		return 0;
	}
	virtual s32 RegisterCustomThreadRoutine()
	{
		return 0;
	}
};

int main(int argc, char* argv[])
{
	if (_opentype == CAMERA_DEVICE_OPENTYPE_GROUPC_SENSOR_FOUR_OVXIF) {
		u32 sensori;

		for (sensori = 0; sensori < 4; sensori++)
		{
			string filename = "testfile_" + std::to_string(sensori) + ".yuv422";
			remove(filename.c_str());
			_parray_pfiletest_index_yuv422[sensori] = fopen(filename.c_str(), "wb");
			if (!_parray_pfiletest_index_yuv422[sensori]) {
				printf("Failed to create output file\r\n");
				return -1;
			}
		}
	}
	s32 ret = 0;
	CameraDeviceCallbackOrin callbackorin;
	ICameraDeviceSession* psession;
	_pcameradevice = ICameraDevice::GetInstance(HAL_CAMERA_VERSION_0_1);
	if ((ret = _pcameradevice->CreateCameraSession(&callbackorin, &psession)) < 0) {
		printf("Camera Device Open fail! ret=%x\r\n", ret);
		return -1;
	}
	printf("sleep 5 seconds\r\n");
	sleep(5);
	psession->Close();
	return 0;
}
