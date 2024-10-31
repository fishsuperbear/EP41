{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/SteerInfo",
  "Namespace": "::ara::chassis9000",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "steerAngle",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "steerAngleRate",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "steerPinionAngle",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "steerPinionAngleRate",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "frontSteerAngle",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "frontSteerAngleRate",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "driverHandOn",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Uint8WithValid",
      "Padding": "2"
    },
    {
      "Name": "driverHandTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "steerTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "motorCurrent",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Uint8WithValid"
    },
    {
      "Name": "personalMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "commandFault",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsStatusMaster",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsStatusSlave",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "2"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "drvrSteerMonrEnaSts",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "handsOffConf",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Uint8WithValid",
      "Padding": "1"
    },
    {
      "Name": "epsMotorTq",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    },
    {
      "Name": "epsMotorTemp",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "ldwWarnSts",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsOperMod",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsAbortFb",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsTqSensSts",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epsSteerAgSensFilr",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerWorkMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "1"
    },
    {
      "Name": "steerCmd",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Float32WithValid"
    }
  ],
  "IsPlane": "true"
}