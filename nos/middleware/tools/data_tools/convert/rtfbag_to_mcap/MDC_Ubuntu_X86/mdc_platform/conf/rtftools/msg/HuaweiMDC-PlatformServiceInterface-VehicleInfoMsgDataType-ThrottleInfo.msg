{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/ThrottleInfo",
  "Namespace": "::mdc::ssm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "throttlePedal",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "throttlePedalRate",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Uint8WithValid",
      "Padding": "2"
    },
    {
      "Name": "driveTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "driveTorqueMax",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "driveTorqueMin",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "driverDesiredTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "engineSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "motorSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "powerTrainReady",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Uint8WithValid"
    },
    {
      "Name": "driveTorqueCtrlAvl",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driveTorqueActv",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "vcuAbortFbk",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "3"
    },
    {
      "Name": "fuelRange",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "socHighVoltBattery",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "sohHighVoltBattery",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "socLowVoltBattery",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "sohLowVoltBattery",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "vcuStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "personalMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "commandFault",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "1"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "frontMotorSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "frontAxleActualTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "rearMotorSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "rearAxleActualTorque",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Float32WithValid"
    },
    {
      "Name": "trailerStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "3"
    }
  ],
  "IsPlane": "true"
}