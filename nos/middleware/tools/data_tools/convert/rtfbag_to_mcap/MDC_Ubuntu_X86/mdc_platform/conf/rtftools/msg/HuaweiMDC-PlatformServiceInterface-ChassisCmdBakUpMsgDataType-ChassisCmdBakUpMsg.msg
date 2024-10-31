{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisCmdBakUpMsgDataType/ChassisCmdBakUpMsg",
  "Namespace": "::ara::chass::cmdbakup",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "source",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "moduleId",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "adsFunction",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "adsWorkStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "longAccelerationCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "longAccelerationValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "longAccelerationState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "longControlMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "jerkMax",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "jerkMin",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "comfortBoundaryUp",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "comfortBoundaryLow",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "longAccelerationPredictCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "rpaEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "comfortBoundaryValid",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "abaCtrlEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "abaLevel",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "secureReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "shutdownReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "suprAvail",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "jerkBrakeReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "motTypeReqMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "idcAebObjSty",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "idcBrkErrSts",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakePedalReqCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "redc",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisCmdBakUpMsgDataType/RedCmd"
    },
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisCmdBakUpMsgDataType/ChassisHeader"
    }
  ],
  "IsPlane": "false"
}