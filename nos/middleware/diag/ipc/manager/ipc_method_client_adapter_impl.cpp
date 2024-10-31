#include "diag/ipc/manager/ipc_method_client_adapter_impl.h"

namespace hozon {
namespace netaos {
namespace diag {

IPCMethodClientAdapterImpl::IPCMethodClientAdapterImpl():
connected_(-1)
,stop_flag_(false)
{
}

IPCMethodClientAdapterImpl::~IPCMethodClientAdapterImpl()
{
    // std::cout << "IPCMethodClientAdapterImpl::~IPCMethodClientAdapterImpl()" << std::endl;
    // Deinit();
}

int32_t IPCMethodClientAdapterImpl::Init(const std::string& service_name)
{
    // std::cout << "IPCMethodClientAdapterImpl::Init() : " << service_name << std::endl;
    std::string req_name = service_name + "_req";
    std::string resp_name = service_name + "_resp";
    msg_req_ipc_ = std::make_unique<msg_line>(req_name.c_str(), ipc::sender);
    msg_resp_ipc_ = std::make_unique<msg_line>(resp_name.c_str(), ipc::receiver);

    checkAvailable_ = std::thread(&IPCMethodClientAdapterImpl::CheckAlive, this);

    // 用于满足立刻调用req的场景，以免发送请求太快，而服务未找到
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return 0;
}

int32_t IPCMethodClientAdapterImpl::Deinit()
{
    // std::cout << "IPCMethodClientAdapterImpl::Deinit()" << std::endl;
    stop_flag_ = true;
    // std::cout << "IPCPathRemove recv_count is" << msg_req_ipc_->recv_count() << std::endl;
    // 仅仅当不存在连接，才清理资源
    if (msg_req_ipc_->recv_count() == 0)
    {
        // std::cout << "IPCPathRemove !!!" << std::endl;
        auto res = IPCPathRemove(default_path);
        if(!res)
        {
            // std::cout << "IPCMethodClientAdapterImpl::Deinit() error." << std::endl;
            return -1;
        }
    }
    
    // 析构函数结束线程
    if (checkAvailable_.joinable()) {
        checkAvailable_.join();
    }
    msg_req_ipc_->disconnect();
    msg_resp_ipc_->disconnect();
    return 0;
}

int32_t IPCMethodClientAdapterImpl::Request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp, const int64_t& timeout_ms)
{
    /*  1. 将req序列化转化为std::vector<uint8_t>
    *   2. 检查通道是否建立，然后发送数据send
    *   3. 开始等待回应，timeout_ms的超时时间，recv阻塞等待
    *   4. 收到resp数据，反序列化，写入resp
    *   5. 程序返回
    */
    // std::cout << "IPCMethodClientAdapterImpl::Request()" << std::endl;
    if (IsMatched() != 0)
    {
        // std::cout << "receiver is not online!" << std::endl;
        return -1;
    }
    auto send_res = msg_req_ipc_->try_send(req.data(), req.size());
    if (!send_res)
    {
        // std::cout << "req data send error!" << std::endl;
        return -1;
    }

    // 等待回复
    ipc::buff_t buf = msg_resp_ipc_->recv(timeout_ms);
    if (buf.empty())
    {
        // std::cout << "resp data receive error!" << std::endl;
        resp.clear();
        return -1;
    }
    resp = CharPointerToVector(static_cast<char*>(buf.data()), buf.size());
    // std::cout << "server resp is :" << static_cast<char*>(buf.data()) << std::endl;

    return 0;
}
    
int32_t IPCMethodClientAdapterImpl::RequestAndForget(const std::vector<uint8_t>& req)
{
    /*  1. 将req序列化转化为std::vector<uint8_t>
    *   2. 检查通道是否建立，然后发送数据send
    *   3. 程序返回
    */
    // std::cout << "IPCMethodClientAdapterImpl::RequestAndForget()" << std::endl;
    if (IsMatched() != 0)
    {
        // std::cout << "receiver is not online!" << std::endl;
        return -1;
    }
    auto send_res = msg_req_ipc_->try_send(req.data(), req.size());
    if (!send_res)
    {
        // std::cout << "req data send error!" << std::endl;
        return -1;
    }
    return 0;
}

int32_t IPCMethodClientAdapterImpl::IsMatched()
{
    return connected_;
}

void IPCMethodClientAdapterImpl::CheckAlive()
{
    // std::cout << "IPCMethodClientAdapterImpl::CheckAlive()" << std::endl;

    // 启动线程一直检查接受者的个数
    while (!stop_flag_)
    {
        // std::cout << "recv_count:" << msg_req_ipc_->recv_count() << std::endl;
        if (msg_req_ipc_->recv_count() >= 1)
        {
            connected_ = 0;
        } else {
            connected_ = -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
