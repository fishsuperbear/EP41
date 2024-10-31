{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ActGacControlCommandSubDataType/ActGacControlCommandMsg",
  "Namespace": "::ara::actgaccontrolcommand",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ActCompensationGacMsgDataType/ActCommonHeader"
    },
    {
      "Name": "socCtrlTimestampSec",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "socCtrlTimestampNsec",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "socCtrlTimestampValid",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacTrqReqVd",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacLongCtrlModeForBrkSys",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacLongCtrlDectostopReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacLongCtrlDriveOff",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacSteerCtrlReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "adcAngReqStA",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "adcVlcSt",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "socGacLongCtrlTargetAccel",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "socGacActTrqReq",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "socGacSteerTorque",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "adcStrAngleReqA",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "socGacLongCtrlAccelCtrlReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "adcLongCtrlTypReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "reserved1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "reserved2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    }
  ],
  "IsPlane": "false"
}