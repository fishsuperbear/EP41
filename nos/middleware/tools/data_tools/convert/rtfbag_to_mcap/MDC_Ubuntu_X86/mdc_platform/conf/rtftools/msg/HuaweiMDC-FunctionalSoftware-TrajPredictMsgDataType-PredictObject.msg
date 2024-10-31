{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/TrajPredictMsgDataType/PredictObject",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "id",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "position",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "theta",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "hasVelocity",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "size",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "polygonPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "trackingTime",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "cls",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "confidence",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "gpsTime",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonTime"
    },
    {
      "Name": "acceleration",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "anchorPoint",
      "Type": "/HuaweiMDC/ApplicationDataType/Point"
    },
    {
      "Name": "boundingBox",
      "Type": "/HuaweiMDC/ApplicationDataType/PointArray"
    },
    {
      "Name": "heightAboveGround",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "positionCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "velocityCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "accelerationCovariance",
      "Type": "/HuaweiMDC/ApplicationDataType/Matrix3d"
    },
    {
      "Name": "lightStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "predictedPeriod",
      "Type": "/HuaweiMDC/StdCppDataType/Double"
    },
    {
      "Name": "trajectory",
      "Type": "/HuaweiMDC/FunctionalSoftware/TrajPredictMsgDataType/TrajInPredictionVector"
    },
    {
      "Name": "intent",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "priority",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "isStatic",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    }
  ],
  "IsPlane": "false"
}