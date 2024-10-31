#include "diag/ipc/manager/ipc_method_server_adapter_impl.h"

namespace hozon {
namespace netaos {
namespace diag {

IPCMethodServerAdapterImpl::IPCMethodServerAdapterImpl():
connected_(false)
,stop_flag_(false)
{
}

IPCMethodServerAdapterImpl::~IPCMethodServerAdapterImpl()
{
    // std::cout << "IPCMethodServerAdapterImpl::~IPCMethodServerAdapterImpl()" << std::endl;
    // Stop();
}

int32_t IPCMethodServerAdapterImpl::Start(const std::string& service_name)
{
    // std::cout << "IPCMethodServerAdapterImpl::Start() : " << service_name << std::endl;
    std::string req_name = service_name + "_req";
    std::string resp_name = service_name + "_resp";
    msg_req_ipc_ = std::make_unique<msg_line>(req_name.c_str(), ipc::receiver);
    msg_resp_ipc_ = std::make_unique<msg_line>(resp_name.c_str(), ipc::sender);
    checkAvailable_ = std::thread(&IPCMethodServerAdapterImpl::CheckAlive, this);
    waitReq_ = std::thread(&IPCMethodServerAdapterImpl::WaitRequest, this);
    return 0;
}

int32_t IPCMethodServerAdapterImpl::Stop()
{
    // std::cout << "IPCMethodServerAdapterImpl::Stop()" << std::endl;
    is_quit__.store(true, std::memory_order_release);
    stop_flag_ = true;
    // std::cout << "IPCPathRemove recv_count is" << msg_resp_ipc_->recv_count() << std::endl;
    // 仅仅当不存在连接，才清理资源
    if (msg_resp_ipc_->recv_count() == 0)
    {
        // std::cout << "IPCPathRemove !!!" << std::endl;
        auto res = IPCPathRemove(default_path);
        if(!res)
        {
            // std::cout << "IPCMethodServerAdapterImpl::Stop() error." << std::endl;
            return -1;
        }
    }

    // 析构函数结束线程
    if (checkAvailable_.joinable()) {
        checkAvailable_.join();
    }

    if (waitReq_.joinable()) {
        waitReq_.join();
    }

    msg_req_ipc_->disconnect();
    msg_resp_ipc_->disconnect();
    return 0;
}

void IPCMethodServerAdapterImpl::RegisterProcess(ProcessFunc func) {
    // std::cout << "IPCMethodServerAdapterImpl::RegisterProcess()" << std::endl;

    process_ = std::move(func);
}

int32_t IPCMethodServerAdapterImpl::WaitRequest()
{
    /*  1. 阻塞等待请求req
    *   2. 调用继承子类的process函数处理请求,若为RR类型，进行3，否则退出本次循环
    *   3. 得到处理结果后将数据发送给客户端
    *   4. 程序返回
    */
    while (!is_quit__.load(std::memory_order_acquire)) {
        // 阻塞等待
        ipc::buff_t buf = msg_req_ipc_->recv(2000);
        if (buf.empty())
        {
            // std::cout << "empty" << std::endl;
            continue;
        }
        auto str = static_cast<char*>(buf.data());
        std::vector<uint8_t> req = CharPointerToVector(str, buf.size());
        std::vector<uint8_t> resp;

        // std::cout << "client req is :" << str << std::endl;

        if (process_ != nullptr) 
        {
            process_(req, resp);
        } else {
            // std::cout << "Call back error: " << std::endl;
            return -1;
        }
        
        // 如果是FF类型，此次数据接收循环结束。
        if (resp.empty())
        {
            // std::cout << "no need response!" << std::endl;
            continue;
        }
        
        if (!connected_)
        {    
            // std::cout << "receiver is not online!" << std::endl;
            return -1;
        }
        auto send_res = msg_resp_ipc_->try_send(resp.data(), resp.size());
        if (!send_res)
        {
            // std::cout << "resp data send error!" << std::endl;
            return -1;
        }
        // std::cout << "resp data send success!" << std::endl;
    }
    // std::cout << "WaitRequest() exit!" << std::endl;
    return 0;
}

void IPCMethodServerAdapterImpl::CheckAlive()
{
    // std::cout << "IPCMethodServerAdapterImpl::CheckAlive()" << std::endl;

    // 启动线程一直检查接受者的个数
    while (!stop_flag_)
    {
        // std::cout << "recv_count:" << msg_resp_ipc_->recv_count() << std::endl;
        if (msg_resp_ipc_->recv_count() >= 1)
        {
            connected_ = true;
        } else {
            connected_ = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
