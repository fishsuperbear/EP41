#include "hal_camera_impl.h"
#include <iostream>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <vector>
/*
* The default thread routine.
*/
#include "default_threadroutine.h"
#include "pipelineconfig_default.h"

s32 CameraDeviceSessionImpl::Internal_RegisterDefaultThreadRoutine()
{
    _threadroutine_handleframe_default = Internal_ThreadRoutine_OutputPipeline_Default;
    _threadroutine_sensornotif_default = Internal_ThreadRoutine_SensorPipeline_Default;
    _threadroutine_blocknotif_default = Internal_ThreadRoutine_BlockPipeline_Default;
    return 0;
}

s32 CameraDeviceSessionImpl::Internal_PrepareLowerLevelConfig_AndHalBasicInfo()
{
    /*
    * First set pipeline config and other lower level members.
    */
    switch (_opentype)
    {
    case CAMERA_DEVICE_OPENTYPE_MULTIROUP_SENSOR_DESAY:
        _numblocks = 3;
        _numsensors = 4;
        _blocktype = CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
        _capturewidth = 1280;
        _captureheight = 960;
        _width = _capturewidth;
        _height = _captureheight;
        Internal_SetPipelineConfig_ToDefault_MultiGroupMain(&_pipelineconfig);
        break;
    default:
        HAL_CAMERA_LOG_ERR("Internal_PrepareLowerLevelConfig_AndHalBasicInfo fail! Unexpected _opentype[%u]\r\n", _opentype);
        return -1;
    }
    /*
    * Secondly set thread routine parray to default.
    */
    u32 blocki;
    u32 sensori;
    for (blocki = 0; blocki < _numblocks; blocki++)
    {
        _parray_threadroutine_blocknotif[blocki] = _threadroutine_blocknotif_default;
        _parray_threadpcontext_blocknotif[blocki] = nullptr;
        for (sensori = 0; sensori < _numsensors; sensori++)
        {
            _parray_threadroutine_handleframe[blocki][sensori] = _threadroutine_handleframe_default;
            _parray_threadpcontext_handleframe[blocki][sensori] = nullptr;
            _parray_threadroutine_sensornotif[blocki][sensori] = _threadroutine_sensornotif_default;
            _parray_threadpcontext_sensornotif[blocki][sensori] = nullptr;
        }
    }
    return 0;
}

s32 CameraDeviceSessionImpl::Internal_Init(ICameraDeviceCallback* i_pcallback)
{
    s32 ret;
    _opentype = i_pcallback->RegisterOpenType();
    _openmode = i_pcallback->RegisterOpenMode();
    /*
    * Set the default thread routine before we call Internal_PrepareLowerLevelConfig_AndHalBasicInfo.
    */
    if ((ret = Internal_RegisterDefaultThreadRoutine()) < 0) {
        HAL_CAMERA_LOG_ERR("Camera Device Session RegisterDefaultThreadRoutine fail! ret=%u\r\n", ret);
        return ret;
    }
    /*
    * Currently, _numblocks, _numsensors, _blocktype, _capturewidth, _captureheight, _width, _height
    * are all set in the following function.
    *
    * The thread routine array is prepared in the following operation but no custom registered.
    */
    if ((ret = Internal_PrepareLowerLevelConfig_AndHalBasicInfo()) < 0) {
        HAL_CAMERA_LOG_ERR("Camera Device Session Internal_PrepareLowerLevelConfig_AndHalBasicInfo fail! ret=%u\r\n", ret);
        return ret;
    }
    /*
    * Just write to the member of CameraDeviceSessionImpl data callback array.
    */
    if ((ret = i_pcallback->RegisterCallback()) < 0) {
        HAL_CAMERA_LOG_ERR("Camera Device Session RegisteCallback fail! ret=%u\r\n", ret);
        return ret;
    }
    /*
    * After prepare all of the open settings, we call ops device_open and pipeline_open of lower level
    * device ops.
    */
    struct hw_video_info_t videoinfo;
    u32 numblocks;
    u32 numsensors;
    if ((ret = _pvideo->ops.device_open(_pvideo, &videoinfo)) < 0) {
        HAL_CAMERA_LOG_ERR("device_open fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("device_open success!\r\n");
    if (_numblocks > videoinfo.numblocks) {
        HAL_CAMERA_LOG_ERR("_numblocks[%u] > videoinfo.numblocks[%u] check fail!\r\n", _numblocks, videoinfo.numblocks);
        return -1;
    }
    numblocks = _numblocks;
    u32 blocki;
    u32 sensori;
    u32 outputi;
    u32 arraynumdatacbs;
    u32 datacbi;
    struct hw_video_blockpipeline_ops_t* pblockpipeline_ops;
    struct hw_video_sensorpipeline_ops_t* psensorpipeline_ops;
    struct hw_video_outputpipeline_ops_t* poutputpipeline_ops;
    struct hw_video_blockinfo_t* pblockinfo;
    struct hw_video_blockpipelineconfig_t* pblockconfig;
    struct hw_video_sensorpipelineconfig_t* psensorconfig;
    CameraDeviceDataCbEnv* pdatacbenv;
    struct hw_video_sensorpipelinedatacbconfig_t* pdatacbconfig;
    /*
    * Set to the data callback info to the lower level pipeline config by CameraDeviceSessionImpl data callback array.
    */
    for (blocki = 0; blocki < numblocks; blocki++)
    {
        pblockinfo = &videoinfo.parrayblockinfo[blocki];
        numsensors = pblockinfo->numsensors_byblock;
        pblockconfig = &_pipelineconfig.parrayblock[blocki];
        for (sensori = 0; sensori < numsensors; sensori++)
        {
            psensorconfig = &pblockconfig->parraysensor[sensori];
            arraynumdatacbs = psensorconfig->datacbsconfig.arraynumdatacbs = _parray_datacbnum[blocki][sensori];
            for (datacbi = 0; datacbi < arraynumdatacbs; datacbi++)
            {
                pdatacbenv = &_parray_datacbenv[blocki][sensori][datacbi];
                pdatacbconfig = &psensorconfig->datacbsconfig.parraydatacbs[datacbi];
                pdatacbconfig->bused = 1;
                pdatacbconfig->type = pdatacbenv->lowerlevelregdatacbtype;
                if (pdatacbconfig->type != HW_VIDEO_REGDATACB_TYPE_CUDA)
                {
                    pdatacbconfig->busecaptureresolution = pdatacbenv->datacbreginfo.busecaptureresolution;
                    pdatacbconfig->customwidth = pdatacbenv->datacbreginfo.customwidth;
                    pdatacbconfig->customheight = pdatacbenv->datacbreginfo.customheight;
                    pdatacbconfig->busecaptureframerate = pdatacbenv->datacbreginfo.busecaptureframerate;
                    pdatacbconfig->customframerate = pdatacbenv->datacbreginfo.customframerate;
                    pdatacbconfig->rotatedegrees = pdatacbenv->datacbreginfo.rotatedegrees;
                    pdatacbconfig->cb = pdatacbenv->lowerleveldatacb;
                    if(_openmode == CAMERA_DEVICE_OPENMODE_SUB){
                        _enable_shm = true;
                    }
                }
                else
                {
                    pdatacbconfig->busecaptureresolution = pdatacbenv->gpudatacbreginfo.busecaptureresolution;
                    pdatacbconfig->customwidth = pdatacbenv->gpudatacbreginfo.customwidth;
                    pdatacbconfig->customheight = pdatacbenv->gpudatacbreginfo.customheight;
                    pdatacbconfig->busecaptureframerate = pdatacbenv->gpudatacbreginfo.busecaptureframerate;
                    pdatacbconfig->customframerate = pdatacbenv->gpudatacbreginfo.customframerate;
                    pdatacbconfig->rotatedegrees = pdatacbenv->gpudatacbreginfo.rotatedegrees;
                    if ((ret = CameraDeviceImpl::GetLowerLevelCudaImgType_ByGpuImageType(pdatacbenv->gpudatacbreginfo.gpuimgtype, &pdatacbconfig->cudaconfig.imgtype)) < 0) {
                        HAL_CAMERA_LOG_ERR("GetLowerLevelCudaImgType_ByGpuImageType fail in Internal_Init!\r\n");
                        return ret;
                    }
                    if ((ret = CameraDeviceImpl::GetLowerLevelInterpolation_ByInterpolation(pdatacbenv->gpudatacbreginfo.interpolation, &pdatacbconfig->cudaconfig.interpolation)) < 0) {
                        HAL_CAMERA_LOG_ERR("GetLowerLevelInterpolation_ByInterpolation fail in Internal_Init!\r\n");
                        return ret;
                    }
                    pdatacbconfig->cudacb = pdatacbenv->lowerlevelcudadatacb;
                }

                // currently always is sync mode
                pdatacbconfig->bsynccb = 1;
                pdatacbconfig->pcustom = pdatacbenv;
            }
            if(_enable_shm)
            {
                _handleFramethreads.push_back(std::thread(processFrame,blocki,sensori,this,pdatacbenv->datacb));
            }
        }
    }
    if(_enable_shm)
    {
        return 0;
    }
    if ((ret = _pvideo->ops.pipeline_open(_pvideo, &_handlepipeline, &_pipelineconfig, &_pblockspipeline_ops)) < 0) {
        HAL_CAMERA_LOG_ERR("pipeline_open fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("pipeline_open success!\r\n");
    /*
    * Before pipeline_start we init the thread routines by CameraDeviceSessionImpl threadroutine array.
    */
    for (blocki = 0; blocki < numblocks; blocki++)
    {
        pblockpipeline_ops = &_pblockspipeline_ops->parrayblock[blocki];
        if (pblockpipeline_ops->bused && (_openmode == CAMERA_DEVICE_OPENMODE_MAIN))
        {
            _vthreadblocknotif.push_back(std::make_unique<std::thread>(_parray_threadroutine_blocknotif[blocki],
                pblockpipeline_ops, _parray_threadpcontext_blocknotif[blocki]));
            numsensors = pblockpipeline_ops->numsensors;
            for (sensori = 0; sensori < numsensors; sensori++)
            {
                psensorpipeline_ops = &pblockpipeline_ops->parraysensor[sensori];
                if (psensorpipeline_ops->bused)
                {
                    _vthreadpipelinenotif.push_back(std::make_unique<std::thread>(_parray_threadroutine_sensornotif[blocki][sensori],
                        psensorpipeline_ops, _parray_threadpcontext_sensornotif[blocki][sensori]));
                    for (outputi = 0; outputi <= HW_VIDEO_SENSORPIPELINE_OUTPUTTYPE_MAX; outputi++)
                    {
                        poutputpipeline_ops = &psensorpipeline_ops->parrayoutput[outputi];
                        if (poutputpipeline_ops->bused)
                        {
                            _vthreadoutput.push_back(std::make_unique<std::thread>(_parray_threadroutine_handleframe[blocki][sensori],
                                poutputpipeline_ops, _parray_threadpcontext_sensornotif[blocki][sensori], outputi));
                        }
                    }
                }
            }
        }
    }
    /*
    * Pipeline start.
    */
    if ((ret = _pvideo->ops.pipeline_start(_pvideo, _handlepipeline)) < 0) {
        HAL_CAMERA_LOG_ERR("pipeline_start fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("pipeline_start success!\r\n");
    return 0;
}

s32 CameraDeviceSessionImpl::Close()
{
    if(_enable_shm)
    {
        for (auto& upthread : _handleFramethreads)
        {
            upthread.join();
        }
        return 0;
    }
    s32 ret;
    if ((ret = _pvideo->ops.pipeline_stop(_pvideo, _handlepipeline)) < 0) {
        HAL_CAMERA_LOG_ERR("pipeline_stop fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("pipeline_stop success!\r\n");
    for (auto& upthread : _vthreadpipelinenotif) {
        if (upthread != nullptr) {
            upthread->join();
            upthread.reset();
        }
    }
    for (auto& upthread : _vthreadblocknotif) {
        if (upthread != nullptr) {
            upthread->join();
            upthread.reset();
        }
    }
    for (auto& upthread : _vthreadoutput) {
        if (upthread != nullptr) {
            upthread->join();
            upthread.reset();
        }
    }
    if ((ret = _pvideo->ops.pipeline_close(_pvideo, _handlepipeline)) < 0) {
        HAL_CAMERA_LOG_ERR("pipeline_close fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("pipeline_close success!\r\n");
    if ((ret = _pvideo->ops.device_close(_pvideo)) < 0) {
        HAL_CAMERA_LOG_ERR("device_close fail! ret=%x\r\n", ret);
        return ret;
    }
    HAL_CAMERA_LOG_UNMASK("device_close success!\r\n");
    return 0;
}

void CameraDeviceSessionImpl::processFrame(int blockidx,int sensoridx,CameraDeviceSessionImpl* context,camera_device_datacb datacb)
{
    context->_consumer_ser_fd[blockidx][sensoridx] = socket(AF_UNIX, SOCK_STREAM, 0);
    if (context->_consumer_ser_fd[blockidx][sensoridx] < 0) {
        return;
    }
    char socket_name[96];
    sprintf(socket_name,SOCKET_NAME,blockidx,sensoridx);
    // set local socket address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_name, sizeof(addr.sun_path) - 1);

    // connect to socket
    if (connect(context->_consumer_ser_fd[blockidx][sensoridx], (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        HAL_CAMERA_LOG_ERR("connect error to %s\n",socket_name);
        return;
    }
    HAL_CAMERA_LOG_INFO("socket connect to %s success.\n",socket_name);

    while(1)
    {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(context->_consumer_ser_fd[blockidx][sensoridx], &readfds);
        struct timeval timeout;
        timeout.tv_sec = 2;  // set timeout 5s
        timeout.tv_usec = 0;
        int ret = select(context->_consumer_ser_fd[blockidx][sensoridx] + 1, &readfds, NULL, NULL, &timeout);
        if(ret==0){//timeout
            HAL_CAMERA_LOG_ERR("get timeout;b:%d,s:%d\n",blockidx,sensoridx);
            continue;
        }
        if(ret<0){//exit
            break;
        }

        // recevied command from server
        char command[64];
        int read_len = read(context->_consumer_ser_fd[blockidx][sensoridx], command, sizeof(command) - 1);
        if (read_len < 0) {
            HAL_CAMERA_LOG_ERR("read error\n");
            exit(1);
        } else if (read_len == 0) {
            HAL_CAMERA_LOG_INFO("connection closed\n");
            exit(1);
        } else {
            command[read_len] = '\0';
            HAL_CAMERA_LOG_INFO("received command: %s\n", command);
        }
        char *token;
        token = strtok(command, ":");
        int data_len = 0;
        char cmd[56];
        if (token != NULL) {
            /* std::cout << "Command: " << token << std::endl; */
            memcpy(cmd,token,strlen(token));
            token = strtok(NULL, ":");
            if (token != NULL) {
                data_len = std::stoi(token);
                /* std::cout << "Data length: " << data_len << std::endl; */
            } else {
                /* std::cerr << "Invalid command format" << std::endl; */
            }
        }

        if (strcmp(command, "bufready") == 0) {
            HAL_CAMERA_LOG_DEBUG("get buf,send unlock;b:%d,s:%d\n",blockidx,sensoridx);
            int shmid = shmget(context->getShmKeyBySensorID(blockidx,sensoridx), SHM_SIZE, 0666|IPC_CREAT);
            if (shmid < 0) {
                HAL_CAMERA_LOG_ERR("shmget error\n");
            }
            void *shmbuf = (void*)shmat(shmid, NULL, 0);
            if (shmbuf == (void*)-1) {
                HAL_CAMERA_LOG_ERR("shmat error\n");
            }
            //call back
            if(datacb){
                CameraDeviceDataCbInfo dataInfo;
                dataInfo.size=data_len;
                hw_video_shmimageheader_t* header = reinterpret_cast<hw_video_shmimageheader_t*>(shmbuf);
                dataInfo.timeinfo.timestamp=header->timestamp;
                dataInfo.pbuff=(void*)((u8*)shmbuf+SHM_HEAD_SIZE);
                switch(blockidx)
                {
                    case 0:
                        dataInfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPA;
                        break;
                    case 1:
                        dataInfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPB;
                        break;
                    case 2:
                        dataInfo.blocktype =CAMERA_DEVICE_BLOCK_TYPE_GROUPC;
                        break;

                }
                dataInfo.sensorindex=sensoridx;
                datacb(&dataInfo);
            }


            if (shmdt(shmbuf) < 0) {
                HAL_CAMERA_LOG_ERR("shmdt error\n");
                break;
            }

            // send "unlock"
            char unlock[] = "unlock";
            int write_len = write(context->_consumer_ser_fd[blockidx][sensoridx], unlock, strlen(unlock));
            if (write_len < 0) {
                HAL_CAMERA_LOG_ERR("write error\n");
            }
        }

    }
    ::close(context->_consumer_ser_fd[blockidx][sensoridx]);
}

int CameraDeviceSessionImpl::getShmKeyBySensorID(int blockidx,int sensoridx){
    int index = (blockidx<<4) | sensoridx;
    switch(index){
        case 0x00:
            return SHM_KEY_BLOCK0_SENSOR0;
        case 0x01:
            return SHM_KEY_BLOCK0_SENSOR1;
        case 0x02:
            return SHM_KEY_BLOCK0_SENSOR2;
        case 0x03:
            return SHM_KEY_BLOCK0_SENSOR3;
        case 0x10:
            return SHM_KEY_BLOCK1_SENSOR0;
        case 0x11:
            return SHM_KEY_BLOCK1_SENSOR1;
        case 0x12:
            return SHM_KEY_BLOCK1_SENSOR2;
        case 0x13:
            return SHM_KEY_BLOCK1_SENSOR3;
        case 0x20:
            return SHM_KEY_BLOCK2_SENSOR0;
        case 0x21:
            return SHM_KEY_BLOCK2_SENSOR1;
        case 0x22:
            return SHM_KEY_BLOCK2_SENSOR2;
        case 0x23:
            return SHM_KEY_BLOCK2_SENSOR3;
        default:
            return -1;
    }
}
