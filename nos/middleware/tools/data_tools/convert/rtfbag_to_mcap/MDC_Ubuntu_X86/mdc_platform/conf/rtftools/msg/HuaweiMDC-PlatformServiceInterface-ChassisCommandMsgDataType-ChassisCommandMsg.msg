{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisCommandMsgDataType/ChassisCommandMsg",
  "Namespace": "::ara::chass::command",
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
      "Name": "steerAngleCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerAngleValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "steerAngleState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerAngleClean",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerAngleMaxLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerAngleRateMaxLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerAngleMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerTorqueSmoothFactor",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerTorqueCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerTorqueValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "steerTorqueState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerTorqueMaxLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerTorqueMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerVibrationCmd",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerVibrationState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerCrvtCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerCrvtValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "steerCrvtState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "steerCrvtGrdCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerFactorSstyCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerForceLmtReqSts",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
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
      "Name": "aebAccelerationCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "aebAccelerationValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "aebAccelerationState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "prefillEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "jerkBrakeEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "ebaCtrlEnable",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "ebaLevel",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driveTorqueCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "driveTorqueValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "driveTorqueState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driveTorqueLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "rapidRespEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakePreferEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakeTorqueCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "brakeTorqueValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "brakeTorqueState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakeTorqueMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakeTorqueLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "brakeLightOnReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "aebBrakeTorqueCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "aebBrakeTorqueValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "aebBrakeTorqueState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "aebBrakeTorqueLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "stopEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driveOffEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakeHoldReq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "emergencyStopEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "emergencyStopEnableValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "epbCmd",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "epbCmdState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "apaDistSpdEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "speedLimit",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "distanceRemain",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "immediateStopEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "rpaEnable",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gearCmd",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisCommandMsgDataType/GearCmd"
    },
    {
      "Name": "gearCmdValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "gearCmdState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gearCmdClean",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakePedalCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "brakePedalValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "brakePedalState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "brakePedalClean",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "throttlePedalCmd",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "throttlePedalValid",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "throttlePedalState",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "throttlePedalClean",
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
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisCommandMsgDataType/RedCmd"
    },
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisCommandMsgDataType/ChassisHeader"
    }
  ],
  "IsPlane": "false"
}