#include <sharedCtx.h>

#include <mutex>

#include "cuda.h"
#include "mps_utils.h"
#include "netacuda/netacuda.h"
#include "tools.h"
#include "zmq_u.hpp"

std::mutex initMutex;
bool initialized = false;
static CUcontext cuda_context;
static bool _use_mps =false;

bool isMpsServerRunning();

s32 neta_cuda_init(u32 i_benablemps) {
    std::unique_lock<std::mutex> lock(initMutex);
    if (i_benablemps) {
        if (!initialized) {
            // zmq socket for message send&recv between processes
            zmq::context_t context;
            zmq::socket_t socket(context, zmq::socket_type::req);
            try{
                int timeout = 5000;
                if(!isMpsServerRunning()){
                    throw error_t();
                }
                socket.setsockopt(ZMQ_CONNECT_TIMEOUT, &timeout, sizeof(int));
                socket.connect("tcp://0.0.0.0:9999");
            }catch (...) {
                HW_NETA_LOG_UNMASK("neta_cuda_init mps error! do nomps init!\n");
                checkCudaErrorsDRV(cuInit(0));
                return 0;
            }

            checkCudaErrorsDRV(cuInit(0));

            if(init_mps_client(socket, cuda_context)){
                _use_mps = true;
                HW_NETA_LOG_UNMASK("neta_init_mps_context success..\n");
            }else{
                HW_NETA_LOG_UNMASK("neta_cuda_init mps error! do nomps init!\n");
            }
            initialized = true;
        } else {
            checkCudaErrorsDRV(cuInit(0));
            if(_use_mps){
                checkCudaErrorsDRV(cuCtxPushCurrent(cuda_context));
            }
        }
    }else{
        cuInit(0);
        HW_NETA_LOG_DEBUG("neta_cuda_init\n");
    }
    return 0;
}

bool isMpsServerRunning()
{
    FILE* pipe = popen("ps -ef | grep mps_server | grep -v grep", "r");
    if (!pipe) {
        std::cerr << "Failed to execute ps command." << std::endl;
        return false;
    }

    char buffer[128];
    bool isRunning = false;

    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        if (strstr(buffer, "mps_server") != nullptr) {
            isRunning = true;
            break;
        }
    }

    pclose(pipe);

    return isRunning;
}