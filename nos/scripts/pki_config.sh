#!/bin/bash

#TODO: need comment out before 1220
mkdir -p /cfg/pki/conf/

if [ -f /app/runtime_service/pki_service/conf/pki_service.yaml.uat ]; then
    cp /app/runtime_service/pki_service/conf/pki_service.yaml.uat /cfg/pki/conf/pki_service.yaml
fi