{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/BrakeInfo",
  "Namespace": "::ara::actcompensation",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "wheelSpeedFl",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/WheelSpeedInfo"
    },
    {
      "Name": "wheelSpeedFr",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/WheelSpeedInfo"
    },
    {
      "Name": "wheelSpeedRl",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/WheelSpeedInfo"
    },
    {
      "Name": "wheelSpeedRr",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/WheelSpeedInfo"
    },
    {
      "Name": "brakePedal",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/Float32WithValid"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/Float32WithValid"
    },
    {
      "Name": "masterMotorSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/Uint32WithValid"
    },
    {
      "Name": "slaveMotorSpeed",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationMsgDataType/Uint32WithValid"
    }
  ],
  "IsPlane": "true"
}