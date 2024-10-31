#!/usr/bin/env bash

user='guest'
password='Hozon2022'
url='http://10.6.133.99:8082/artifactory'
thirdPartyPath='.'

if [ ! -d ~/.jfrog ]; then
    mkdir -p ~/.jfrog
fi

if [ ! -f ~/.jfrog/jfrog-cli.conf.v5 ]; then
    cp tools/jfrog/jfrog-cli.conf.v5 ~/.jfrog/jfrog-cli.conf.v5
fi

currThirdParty=$(python3 -c "import sys, json;
with open('version.json','r') as f: print(json.load(f)['ORIN']['EP41']['third_party'])")

echo "current third_party in version.json is $currThirdParty"


TPVersionInfo=''
if [ -f $thirdPartyPath/third_party/version.json ]; then
    TPVersionInfo=$(python3 -c "import sys, json;
with open('$thirdPartyPath/third_party/version.json','r') as f: print(json.load(f)['releaseVersion'])")
fi

if [[ "$TPVersionInfo" != "$currThirdParty" ]]; then
    echo "current third_party in third_party/version.json is $TPVersionInfo"
    ./tools/jfrog/jfrog rt dl --flat --user=$user --password=$password --url=$url EP40_MDC_TP/build/$currThirdParty*
    echo "tar -xf $currThirdParty.tar.gz "
    rm -rf $thirdPartyPath/third_party && tar -xf $currThirdParty.tar.gz -C $thirdPartyPath && rm -f $currThirdParty.tar.gz
fi


# add cuda to third_party_stable path 
if [ ! -d $thirdPartyPath/third_party_stable ]; then
    echo "Creating directory: $thirdPartyPath/third_party_stable"
    mkdir -p $thirdPartyPath/third_party_stable
else
    echo "Directory already exists: $thirdPartyPath/third_party_stable"
fi

currCudaVersion=$(python3 -c "import sys, json;
with open('version.json','r') as f: print(json.load(f)['ORIN']['EP41']['CUDA'])")

echo "current cuda in version.json is $currCudaVersion"

cudaVersionInfo=''
if [ -f $thirdPartyPath/third_party_stable/cuda/version.json ]; then
    cudaVersionInfo=$(python3 -c "import sys, json;
with open('$thirdPartyPath/third_party_stable/cuda/version.json','r') as f: print(json.load(f)['releaseVersion'])")
fi

if [[ "$cudaVersionInfo" != "$currCudaVersion" ]]; then
    echo "current version in third_party_stable/cuda/version.json is $cudaVersionInfo"
    ./tools/jfrog/jfrog rt dl --flat --user=$user --password=$password --url=$url nvidia/cuda/$currCudaVersion*
    echo "tar -xf $currCudaVersion.tar.gz "
    rm -rf $thirdPartyPath/third_party_stable/cuda && tar -xf $currCudaVersion.tar.gz -C $thirdPartyPath/third_party_stable/ && rm -f $currCudaVersion.tar.gz
fi

# 将third_party_stable的cuda库链接到third_party下
if [ ! -d $thirdPartyPath/third_party/x86_2004/cuda ]; then
    echo "add link to x86_2004/cuda"
    cd $thirdPartyPath/third_party/x86_2004 && ln -s ../../third_party_stable/cuda cuda && cd -
fi

if [ ! -d $thirdPartyPath/third_party/orin/cuda ]; then
    echo "add link to orin/cuda"
    cd $thirdPartyPath/third_party/orin && ln -s ../../third_party_stable/cuda cuda && cd -
fi

# add tensorrt to third_party_stable path 
currTensorRTVersion=$(python3 -c "import sys, json;
with open('version.json','r') as f: print(json.load(f)['ORIN']['EP41']['TensorRT'])")

echo "current tensorrt in version.json is $currTensorRTVersion"

tensorrtVersionInfo=''
if [ -f $thirdPartyPath/third_party_stable/tensorrt/version.json ]; then
    tensorrtVersionInfo=$(python3 -c "import sys, json;
with open('$thirdPartyPath/third_party_stable/tensorrt/version.json','r') as f: print(json.load(f)['releaseVersion'])")
fi

if [[ "$tensorrtVersionInfo" != "$currTensorRTVersion" ]]; then
    echo "current version in third_party_stable/tensorrt/version.json is $tensorrtVersionInfo"
    ./tools/jfrog/jfrog rt dl --flat --user=$user --password=$password --url=$url nvidia/tensorrt/$currTensorRTVersion*
    echo "tar -xf $currTensorRTVersion.tgz "
    rm -rf $thirdPartyPath/third_party_stable/tensorrt && tar -xf $currTensorRTVersion.tgz -C $thirdPartyPath/third_party_stable/ && rm -f $currTensorRTVersion.tgz
fi

if [ ! -d $thirdPartyPath/third_party/x86_2004/tensorrt ]; then
    echo "add link to x86_2004/tensorrt"
    cd $thirdPartyPath/third_party/x86_2004 && ln -s ../../third_party_stable/tensorrt tensorrt && cd -
fi

if [ ! -d $thirdPartyPath/third_party/orin/tensorrt ]; then
    echo "add link to orin/tensorrt"
    cd $thirdPartyPath/third_party/orin && ln -s ../../third_party_stable/tensorrt tensorrt && cd -
fi


