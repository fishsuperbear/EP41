#include <fstream>

#include "adf-lite/include/ds/builtin_types.h"
#include "adf-lite/include/writer.h"
#include "fisheye_datatype.h"
#include "img_zerocopy_executor.h"

IdlCudaExecutor::IdlCudaExecutor() {}

IdlCudaExecutor::~IdlCudaExecutor() {}

int32_t IdlCudaExecutor::AlgInit() {

    NODE_LOG_INFO << "Init IdlCudaExecutor.";
    // RegistAlgProcessFunc("replay_cam", std::bind(&IdlCudaExecutor::NVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("replay_cam", std::bind(&IdlCudaExecutor::DumpNVSCamProcess, this, std::placeholders::_1));
    RegistAlgProcessFunc("replay_cam_idl", std::bind(&IdlCudaExecutor::NVSCamProcess, this, std::placeholders::_1));

    return 0;
}

void IdlCudaExecutor::AlgRelease() {
    NODE_LOG_INFO << "Release IdlCudaExecutor.";
}

int32_t IdlCudaExecutor::DumpNVSCamProcess(Bundle* input) {
    cam_idx++;

    std::shared_ptr<NvsImageCUDA> camera_0_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_0"));
    if (camera_0_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_0.";
        return -1;
    }

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_0"), cam_idx, camera_0_ptr);
    }

    freq_checker.say("/soc/zerocopy/camera_0");
    std::shared_ptr<NvsImageCUDA> camera_1_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_1"));
    if (camera_1_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_1.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_1");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_1"), cam_idx, camera_1_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_3_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_3"));
    if (camera_3_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_3.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_3");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_3"), cam_idx, camera_3_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_4_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_4"));
    if (camera_4_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_4.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_4");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_4"), cam_idx, camera_4_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_5_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_5"));
    if (camera_5_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_5.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_5");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_5"), cam_idx, camera_5_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_6_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_6"));
    if (camera_6_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_6.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_6");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_6"), cam_idx, camera_6_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_7_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_7"));
    if (camera_7_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_7.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_7");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_7"), cam_idx, camera_7_ptr);
    }

    return 0;

    std::shared_ptr<NvsImageCUDA> camera_8_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_8"));
    if (camera_8_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_8.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_8");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_8"), cam_idx, camera_8_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_9_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_9"));
    if (camera_9_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_9.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_9");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_9"), cam_idx, camera_9_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_10_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_10"));
    if (camera_10_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_10.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_10");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_10"), cam_idx, camera_10_ptr);
    }

    std::shared_ptr<NvsImageCUDA> camera_11_ptr = std::static_pointer_cast<NvsImageCUDA>(input->GetOne("/soc/zerocopy/camera_11"));
    if (camera_11_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_11.";
        return -1;
    }
    freq_checker.say("/soc/zerocopy/camera_11");

    if (dump_file == true) {
        ImageDumpFile(std::string("camera_11"), cam_idx, camera_11_ptr);
    }

    return 0;
}

int32_t IdlCudaExecutor::NVSCamProcess(Bundle* input) {
    cam_idx++;

    BaseDataTypePtr camera_0_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_0"));
    if (camera_0_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv idl camera_0.";
        return -1;
    }

    freq_checker.say("/idl/camera_0");
    BaseDataTypePtr camera_1_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_1"));
    if (camera_1_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv idl camera_1.";
        return -1;
    }
    freq_checker.say("/idl/camera_1");

    BaseDataTypePtr camera_3_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_2"));
    if (camera_3_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_2.";
        return -1;
    }
    freq_checker.say("/idl/camera_2");

    BaseDataTypePtr camera_4_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_4"));
    if (camera_4_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_4.";
        return -1;
    }
    freq_checker.say("/idl/camera_4");

    BaseDataTypePtr camera_5_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_5"));
    if (camera_5_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_5.";
        return -1;
    }
    freq_checker.say("/idl/camera_5");

    BaseDataTypePtr camera_6_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_6"));
    if (camera_6_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_6.";
        return -1;
    }
    freq_checker.say("/idl/camera_6");

    BaseDataTypePtr camera_7_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_7"));
    if (camera_7_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_7.";
        return -1;
    }
    freq_checker.say("/idl/camera_7");

    BaseDataTypePtr camera_8_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_8"));
    if (camera_8_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_8.";
        return -1;
    }
    freq_checker.say("/idl/camera_8");

    BaseDataTypePtr camera_9_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_9"));
    if (camera_9_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_9.";
        return -1;
    }
    freq_checker.say("/idl/camera_9");

    BaseDataTypePtr camera_10_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_10"));
    if (camera_10_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_10.";
        return -1;
    }
    freq_checker.say("/idl/camera_10");

    BaseDataTypePtr camera_11_ptr = std::static_pointer_cast<BaseData>(input->GetOne("/idl/camera_11"));
    if (camera_11_ptr == nullptr) {
        NODE_LOG_INFO << "Fail to recv camera_11.";
        return -1;
    }
    freq_checker.say("/idl/camera_11");

    return 0;
}

void IdlCudaExecutor::WriteFile(const std::string& name, uint8_t* data, uint32_t size) {
    std::ofstream of(name);

    if (!of) {
        NODE_LOG_ERROR << "Fail to open " << name;
        return;
    }

    of.write((const char*)data, size);
    of.close();
    NODE_LOG_INFO << "Succ to write " << name;

    return;
}

void IdlCudaExecutor::ImageDumpFile(const std::string& file_name, int index, std::shared_ptr<NvsImageCUDA> packet) {
    if (index % 10 != 0) {
        return;
    }

    uint8_t* local_ptr = (uint8_t*)malloc(packet->size);

    /* Instruct CUDA to copy the packet data buffer to the target buffer */
    uint32_t cuda_rt_err = cudaMemcpy(local_ptr, packet->cuda_dev_ptr, packet->size, cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuda_rt_err) {
        NODE_LOG_ERROR << "Failed to issue copy command, ret " << packet->cuda_dev_ptr;
        return;
    }

    std::string dump_name = file_name + "_" + std::to_string(index) + ".data";
    WriteFile(dump_name, local_ptr, packet->size);

    free(local_ptr);
}
