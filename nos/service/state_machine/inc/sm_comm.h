#ifndef HOZON_AP_SM_COMM_H
#define HOZON_AP_SM_COMM_H
#include <memory>
#include <thread>
#include <chrono>
#include "adf/include/log.h"
#include "algdata/imu.h"
#include "algdata/egohmi.h"
#include "algdata/parkinglot.h"
#include "algdata/state_machine_frame.h"

/*
    ADCS11_Parking_WorkSts：
    0x0：off
    0x1：Standby(通过开关进入PA)
    0x2：TBA Enable
    0x3：Parking
    0x4：Cruising（HPP）
    0x5：Map building
    0x6：Localization
    0x7：Complete
    0x8：Suspend
    0x9：Abort
    0xA：Failure
     off = 0,                    //待机
    Standby,                    //泊车
    Standby_Searching,          //泊车车位搜索
    TBA_Enable,                 //循迹倒车
    Parking,                    //停车
    Cruising_HPP,               //记忆泊车巡航
    Map_Building,               //建立地图
    Localization,               //定位
    Complete,                   //完成
    Suspend,                    //暂停
    Abort,                      //终止
    Failure,                    //失败
    Cruising_AVP,               //代客泊车巡航
    Invaild                     //无效   
*/
enum class Parking_Work_Status: uint8_t {
    off = 0,
    Standby,//通过AVM界面开关进入PA选项
    TBA_Enable,//对应can信号表中的TBA standby
    Parking,
    Cruising_HPP,
    Map_Building,
    Localization,
    Complete,
    Suspend,
    Abort,
    Failure,
    Cruising_AVP,
    RESET,
    Invaild
};

enum class ParkingMode :uint8_t {
    init = 0,
    FAPA,       //融合自动泊车辅助
    RPA,        //遥控泊车
    DAPA,       //自定义自动泊车，选择泊车
    TBA,        //循迹倒车
    LAPA_Mapping,//记忆泊车建图
    LAPA,       //记忆泊车
    AVP,        //代客泊车
    ISM,         //智能召唤
    NTPPickUp,   //记忆泊车接驾
    Localization, // 定位
    Back_Location // 后台定位
};

//APA_FunctionMode---ADCS4_APA_FunctionMode
enum class APA_FunMode : uint8_t{
    NO_DEF = 0,
    ParkingIn, //泊入
    ParkingOut, //泊出
    ChoiceParkingIn,//选择泊车
    Traction //寻迹倒车
};

//RPA_FunctionMode---ADCS4_RPA_FunctionMode
enum class RPA_FunMode : uint8_t{
    NO_DEF = 0,
    ParkingIn, //泊入
    ParkingOut, //泊出
    ChoiceParkingIn,//选择泊车
    LineCtrl //线控
};

enum class PncState: uint8_t {
    PNC_RUN_STOP =0 ,   //停止
    PNC_RUN_START,      //运行中
    PNC_RUN_CONFIG,     //正在初始化
    PNC_RUN_PARKING,    //泊车中
    PNC_RUN_PAUSE,      //挂起
    PNC_RUN_QUIT,       //退出
    PNC_RUN_FINISH,     //结束
    PNC_RUN_COUNT,      //重规划
    PNC_RUN_ERROR       //出错
};

enum class ControlSatus: uint8_t {
    init = 0,    //默认状态，一般为系统为开启时发送此状态
    running,        //系统已激活并准备接受下一步指令
    parking_out_controlling,    //泊出控制中
    parking_in_controlling,     //泊入控制中
    quit_parking_control,       //HMI触发(HMI介入引发控制退出）
    brake_interval,             //刹车介入，挂起系统
    steering_interval,          //方向盘介入，挂起系统
    waiting_obstacle_disappear, //等待障碍物消失
    control_failure,        //执行机构故障（执行机构故障引发泊车失败）
    parking_in_failed,      //泊入失败
    parking_in_finished,    //泊入结束
    parking_out_finished,   //泊出结束
    during_in_planning,     //触发二次规划
    posture,                //OK-表示车辆已经到达泊车结束位姿，与FusionParkingSlot有关（车辆已到达结束点）
    planning_out_failed,    //泊出失败
    reset = 255      //重置中
};
/*
    0x00:Default（默认状态，没有任何改变时发送此命令）
    0x01:SystemOn（激活模块）
    0x02:ParkingInAutopilot（巡航控制，一般用于记忆泊车时的巡航阶段）
    0x03:ParkingInControl（泊入控制）
    0x04:ParkingOutSearch（泊出控制前，搜索库位）
    0x05:QuitControl（退出控制，可用于巡航阶段，泊入阶段和泊出阶段）
    0x08:ParkingOutControl（泊出控制）
    0x09:ParkingOutAutopilot（泊出完成，开始巡航并到达停车点）
    0x0a：LeftParkingOutControl
    0x0b: RightParkingOutControl
    0x0c：ForwardParkingOutControl（APA向前泊出）
    0x0d:  BackParkingOutControl（APA向后泊出）
    0x0e:   brake (RPA刹停)
    0xof：RecoverControl
    0x10：Forward（RPA向前直行）
    0x11：Back  （RPA向后直行）
*/

enum class CommandCtl: uint8_t{
    Default = 0,
    SystemOn,
    ParkingInAutopilot,
    ParkingIntControl,
    ParkingOutSearch,
    QuitControl,
    ParkingOutControl,
    ParkingOutAutopilot,
    LeftParkingOutControl,
    RightParkingOutControl,
    ForwardParkingOutControl,
    BackParkingOutControl,
    Brake,
    RocoverControl,
    Forward,
    Back
};

//fct_state
/*
    0x0:PNC_RUN_STOP（停止）
    0x1:PNC_RUN_PARKSTART（泊车开始）
    0x2:PNC_RUN_CONFIG（正在初始化）
    0x3:PNC_RUN_PARKING（泊车中）
    0x4:PNC_RUN_PAUSE（挂起）
    0x5:PNC_RUN_QUIT（退出）
    0x6:PNC_RUN_PARKFINISH（泊车结束）
    0x7:PNC_RUN_COUNT（重规划）
    0x8:PNC_RUN_ERROR（出错）
    0x9:PNC_RUN_CRUISESTART（巡航开始）
    0xA:PNC_RUN_CRUISING（巡航中）
    0xB:PNC_RUN_CRUISEFINISH(巡航结束)
    0xC:PNC_RUN_STRAIGHTCONTROL(直行控制)
    0xD:PNC_RUN_STRAIGHTBRAKE(直行刹停)
    0xE:PNC_RUN_TBASTART
    0xF:PNC_RUN_TBACONTROL
    0x10:PNC_RUN_TBAFINISH
    0x11:PNC_RUN_NNSSTART
    0x12:PNC_RUN_NNSCONTROL
    0x13:PNC_RUN_NNSFINISH
*/

#define     PNC_RUN_STOP        0x0
#define     PNC_RUN_PARKSTART   0x1
#define     PNC_RUN_CONFIG      0x2
#define     PNC_RUN_PARKING     0x3
#define     PNC_RUN_PAUSE       0x4
#define     PNC_RUN_QUIT        0x5
#define     PNC_RUN_PARKFINISH  0x6
#define     PNC_RUN_COUNT       0x7
#define     PNC_RUN_ERROR       0x8
#define     PNC_RUN_CRUISESTART 0x9
#define     PNC_RUN_CRUISING    0xA
#define     PNC_RUN_CRUISEFINISH 0xB
#define     PNC_RUN_STRAIGHTCONTROL 0xC
#define     PNC_RUN_STRAIGHTBRAKE   0xD
#define     PNC_RUN_TBASTART    0xE
#define     PNC_RUN_TBACONTROL  0xF
#define     PNC_RUN_TBAFINISH   0x10
#define     PNC_RUN_NNSSTART    0x11
#define     PNC_RUN_NNSCONTROL  0x12
#define     PNC_RUN_NNSFINISH   0x13
#define     PNC_RUN_INVERT      0x17


/*pnc_run_state 枚举值定义
 0-PNC_RUN_STOP（停止）
 1-PNC_RUN_PARKSTART（泊车开始）
 2-PNC_RUN_CONFIG（正在初始化）
 3-PNC_RUN_PARKING（泊车中）
 4-PNC_RUN_PAUSE（挂起）
 5-PNC_RUN_QUIT（退出）
 6-PNC_RUN_PARKFINISH（泊车结束）
 7-PNC_RUN_COUNT（重规划）
 8-PNC_RUN_ERROR（出错）
9-PNC_RUN_CRUISESTART（巡航开始）
10-PNC_RUN_CRUISING（巡航中）
11-PNC_RUN_CRUISEFINISH(巡航结束)
12-PNC_RUN_STRAIGHTCONTROL(直行控制)
13-PNC_RUN_STRAIGHTBRAKE(直行刹停)

*/
struct module_status
{
    AlgWorkingStatus common_status;//对感知、slam、location有效
    AlgPNCControlState pnc_state;//只有对pnc规控模块有效 
};

//底盘信号结构体
struct ChasisSignal {

    //VCU
    uint8_t VCU_ReadySts;
    uint8_t VCU_ActGearPosition;//档位信息
    uint8_t VCU_Real_ThrottlePosition;
    uint8_t VCU_Real_ThrottlePos_Valid;
    uint8_t VCU_APA_Response;
    //0x0：Normal
    //0x1：Over speed
    //0x2：Driver intervene
    //0x3：Can not change gear 
    //0x4：vehicle block
    uint8_t VCU_APA_FailCode; 
    uint8_t VCU5_READYLightSts;//(对应结构体中存在，可以进行编辑)运动准备就绪指示灯  0x0：Lamp OFF    0x1：Lamp ON
    //uint8_t VCU5_EmuEngineSts;//模拟（仿真emulated）的引擎状态，用于start控制逻辑。0x0：Engine Stop   0x1：Engine Running

    //stear and vehicle speed
    uint8_t SteeringAngle;
    float ESC_VehicleSpeed;//车速
    uint8_t ESC_VehicleSpeedValid;//车速有效状态
    uint8_t ESC_BrakePedalSwitchStatus;//制动踏板 0x0：no press(( 制动踏板未被踩下)  0x1：press( 制动踏板被踩下)
    uint8_t ESC_BrakePedalSwitchValid;//刹车踏板被踩下有效状态

    //car body status
    uint8_t BCM_FLDrOpn;//左前门开启
    uint8_t BCM_FRDrOpn;//右前门开启
    uint8_t BCM_RLDrOpn;//左后门开启
    uint8_t BCM_RRDrOpn;//右后门开启
    uint8_t BCM_TGOpn;//尾门开启
    uint8_t BCM_HodOpen;//引擎盖开启
    uint8_t BCM_DrvSeatbeltBucklesta;//驾驶员安全带状态


    //0x0：No Request 0x1：Parking in 0x2：Parking out 
    //0x3：choice parking in 0x4：FAPA on 0x5：RPA on 
    //0x6：HPP on 0x7：AVP on 0x8：ISM on 0x9：TAB on
    uint8_t CDCS11_APA_FunctionMode; //泊车功能模式

    uint8_t CDCS11_RPASw; //RPA软开关

    uint8_t CDCS11_HPAGuideSW; //开关，用户选择是否尝试记忆泊车学习
    uint8_t CDCS11_HPAPathwaytoCloudSW; //记忆泊车路线学习完成，是否上传开关
    uint8_t CDCS11_PathlearningSw; //记忆泊车路线学习开关
    
    /*APA软开关 取值定义
    0x0：No Request  
    0x1：Request PA on  
    0x2：Request PA off  
    0x3： Reserve    
    */
    uint8_t CDCS11_PASw; //APA软开关 

    uint8_t CDCS11_PA_Recover; //IHU发出恢复泊车指令

    uint8_t CDCS11_ParkingInReq; //开始泊入开关
    uint8_t CDCS11_ParkingOutReq; //开始泊出开关

    uint8_t CDCS11_RPA_FunctionMode; //遥控泊车功能模式

    uint8_t CDCS11_HPASw; //开启记忆泊车开关
    uint8_t CDCS11_HPApreparkingSw; //记忆泊车快捷开关
    uint8_t CDCS11_HPAPathwaytoCloud_WorkSts; //路线上传开关状态
    uint8_t CDCS11_ParkingoutSlot_Type; //自选车位类型
    uint8_t CDCS11_ParkingoutSlot_Dire; //自选车位方向
   
    uint8_t CDCS11_OptionalSlot_Type;//可选车位类型
    uint8_t CDCS11_OptionalSlot_Dire;//可选车位方向
    uint8_t CDCS11_location_sw;//定位开关
    uint8_t CDCS11_Pathrefresh;
    uint8_t CDCS11_SelectSlotID;//选择车位号
    uint8_t CDCS11_TrackreverseSW;//循迹倒车开关

    uint8_t BDCS1_PowerManageMode;//电源模式管理信号
    uint8_t BTM1_SecuritySts;
    uint8_t BTM1_PhoneBatSts;
    uint8_t BTM1_RemoteIntModSel;
    uint8_t BTM1_SelectSlotID;
    uint8_t BTM1_Retry;
    uint8_t BTM1_Fixslot;
    uint8_t BTM1_parkingoutSlot_Dire;
    uint8_t BTM1_parkingoutSlotType;
    uint8_t BTM1_Recover;
    uint8_t BTM1_ParkingReq;//开始泊车信息
    uint8_t BTM1_FunctionMode;
    uint8_t BTM1_OptionalSlot_Dire;//蓝牙遥控泊车方向
    uint8_t BTM1_OptionalSlotType;//蓝牙遥控泊车类型
    uint8_t BTM1_RollingCounter;
    uint8_t BTM1_RemoteParkReq;
    uint8_t BTM1_Movectrl;//遥控泊车前进后退信号
    uint8_t ESC_ApaStandStill;//判断车辆静止信号
    uint8_t BDCS1_PowerMode;//车辆低压电源模式
    uint8_t BTM2_ConnectSts;//蓝牙连接状态
    uint8_t TBOX2_RemoteHPP;//远程进入HPP
    uint8_t CDCS11_tryHPP;//记忆泊车试一试按键信号
    uint8_t TBOX2_RemoteParkReq;//手机端泊车请求
    uint8_t CDCS11_AVMSw;//AVM软开关
    uint8_t CDCS11_learnpath_St;
};

struct APA2Chassis{
    uint8_t ADCS11_PA_WorkSts;
    uint8_t ADCS4_APA_FunctionMod;
    uint8_t ADCS4_PA_failinfo;
    uint8_t ADCS4_text;
    uint8_t ADCS8_PA_warninginfo;
    uint8_t ADCS4_AVM_vedioReq;
    uint8_t ADCS4_PA_ParkBarPercent;
    uint8_t ADCS11_TurnLampReq;
    uint8_t ADCS4_parking_time;
    uint8_t ADCS4_ParkingswithReq;
    uint8_t ADCS4_TractionswithReq;
    uint8_t ADCS4_Slotavaliable;
    uint8_t ADCS4_RPA_FunctionMode;
    uint8_t ADCS11_PA_Recover;//APA/RPA泊车恢复指令
    uint8_t ADCS11_Currentgear;//泊车当前档位信息
    uint8_t ADCS4_HPA_FunctionMode;
    uint8_t ADCS4_APA_FunctionMode;
    uint8_t ADCS4_ParkingTime;//RPA模式下的停车计时
    uint8_t ADCS4_HPA_failinfo;
    uint8_t ADCS11_HPA_Path_exist;
    uint8_t ADCS11_HPA_Pathavaliable_id1;
    uint8_t ADCS11_HPA_PathlearningSt;
    uint8_t ADCS11_HPAPathlearning_WorkSts;
    uint8_t ADCS11_HPAGuidSts;
    uint8_t ADCS11_Mapbuilding_distance;
    uint8_t ADCS11_Parking_WorkSts;
    uint8_t ADCS11_HPA_backto_start;
    uint8_t ADCS11_PA_lampreq;//双闪灯
};

struct ParkingTask {
    bool valid{};
    std::chrono::time_point<std::chrono::system_clock> begin_time, end_time;
    uint8_t  suspend_counter{};
    uint8_t switchMode_Counter{};//行车和泊车模式切换计时
    ParkingMode mode{};
    APA_FunMode APA_funmode{};
    RPA_FunMode RPA_funmode{};
    Parking_Work_Status last_status{};
    Parking_Work_Status now_state{};
    // APA2Chassis apa2chassis;
    uint8_t fail_code{};
    uint8_t CDCS11_ParkingoutSlot_Dire{};
    bool TBA_valid{};
};


    int signal_cb(std::shared_ptr<ChasisSignal> p_newSample, 
    std::shared_ptr<AlgStateMachineFrame> pnc_frame,
    std::shared_ptr<AlgStateMachineFrame> perception_frame,
    std::shared_ptr<APA2Chassis>  _apa2chassis,
    module_status module_pnc, 
    module_status module_perception, 
    std::shared_ptr<AlgParkingLotOutArray> pParkingLotArray,std::shared_ptr<AlgEgoHmiFrame> pEgoHmi);
    int state_dump( );
    uint64_t get_time_us();
    bool timeout_func(uint64_t _SetTimeOut);
    int Clear_TimeOutFuncGlobalValue();

    int HPP_ResetFunction(std::shared_ptr<AlgStateMachineFrame>  perception_frame,
    std::shared_ptr<AlgStateMachineFrame> pnc_frame,
    module_status module_perception, 
    module_status module_pnc,std::shared_ptr<APA2Chassis> _apa2chassis_patameter);
    
    int Cruising_HPP_ResetFunction(std::shared_ptr<AlgStateMachineFrame>  perception_frame,
    std::shared_ptr<AlgStateMachineFrame> pnc_frame,
    module_status module_perception,module_status module_pnc);
    
    int convertWarningInfoToTBAInfo(module_status module_pnc_patameter,module_status module_perception_patameter,std::shared_ptr<AlgStateMachineFrame> & slam_frame);

    void set_mapid(int32_t id);
    int32_t get_mapid();


#endif //HOZON_AP_SM_H

