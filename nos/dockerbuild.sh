#!/bin/bash
for i in $*; do ARGS+="$i "; done
RUN_UID=$(id | awk -F'[=(]' '{print $2}')

# 定义Harbor仓库地址和API版本
HARBOR_HOST="10.0.8.143:7201"
API_VERSION="v2.0"

# 定义Harbor用户名和密码
HARBOR_USER="soc"
HARBOR_PASSWORD="Hozon12345"

# 定义要查询的镜像
PROJECTS="soc"
if [[ "${ARGS}" == *"mdc-llvm"* ]]; then
    REPOSITORY="mdc_sdk_0228"
else
    REPOSITORY="orin_sdk"
fi
TAG="latest"

IMAGES_NAME=$HARBOR_HOST/$PROJECTS/$REPOSITORY:$TAG

function update_images() {
    echo $HARBOR_PASSWORD | docker login --username $HARBOR_USER --password-stdin $HARBOR_HOST >/dev/null 2>&1

    RESPONSE=$(curl \
        -u ''$HARBOR_USER':'$HARBOR_PASSWORD'' \
        -H "Content-Type: application/json" \
        -s -X GET 'http://'$HARBOR_HOST'/api/'$API_VERSION'/projects/'$PROJECTS'/repositories/'$REPOSITORY'/artifacts?page=1&page_size=10&with_tag=true&with_label=false&with_scan_overview=false&with_signature=false&with_immutable_status=false&with_accessory=false' \
        -H 'accept: application/json' \
        -H 'X-Accept-Vulnerabilities: application/vnd.scanner.adapter.vuln.report.harbor+json; version=1.0' |
        jq '.[] | select(.tags[].name == "'$TAG'") | select(.extra_attrs != null) | .extra_attrs.created')

    if [ -z "$RESPONSE" ]; then
        echo -e "\033[32mThe docker registry was not found images: '$IMAGES_NAME', please check if parameters are configured correctly...\033[0m"
        exit 1
    fi

    if [ "$(docker images -q $IMAGES_NAME 2>/dev/null)" == "" ]; then
        echo -e "\033[32m$IMAGES_NAME was not found locally, pulling image from Docker registry...\033[0m"
        docker pull $IMAGES_NAME
    else
        CREATETIME=$(date -d $(echo $RESPONSE | sed 's/^"\|"$//g') +%s)
        #echo $CREATETIME
        LOCALTIME=$(date -d $(docker inspect -f '{{.Created}}' $IMAGES_NAME) +%s)
        #echo $LOCALTIME
        if [ $CREATETIME -gt $LOCALTIME ]; then
            echo -e "\033[32mThe repository image has been updated, deleting the local image and pulling the latest image from the repository...\033[0m"
            docker rmi -f $IMAGES_NAME
            docker pull $IMAGES_NAME
        else
            echo -e "\033[32mThe local image is the latest version.\033[0m"
        fi
    fi
}

function check_registries() {
    CONTENT='"insecure-registries": ["http://'$HARBOR_HOST'"]'

    if [ ! -e /etc/docker/daemon.json ]; then
        sudo touch /etc/docker/daemon.json
    fi

    if [ $(jq '.["insecure-registries"] | index("http://'$HARBOR_HOST'") != null' /etc/docker/daemon.json) = false ]; then
        echo -e "\033[32mDocker repository address configuration not found in /etc/docker/daemon.json, adding now...\033[0m"

        if [ ! -s /etc/docker/daemon.json ]; then
            sudo sh -c "echo '{' >> /etc/docker/daemon.json"
            sudo sh -c "echo \"  $CONTENT\" >> /etc/docker/daemon.json"
            sudo sh -c "echo '}' >> /etc/docker/daemon.json"
        else
            sudo sed -i '$s/}/,\n  "insecure-registries": \["http:\/\/'$HARBOR_HOST'"\]\n}/' /etc/docker/daemon.json
        fi

        sudo sh -c "cat /etc/docker/daemon.json | jq '.' > /etc/docker/new_daemon.json"
        sudo mv /etc/docker/new_daemon.json /etc/docker/daemon.json
        sudo systemctl restart docker.service
        echo -e "\033[32mInsecure-registries configuration detected and added successfully. Restarting Docker service...\033[0m"
    fi
}

function check_tools() {
    if ! dpkg -s jq &>/dev/null; then
        echo -e "\033[32mjq not found, installing now...\033[0m"
        sudo apt-get install jq -y
    fi

    if [ $(getent group docker | grep -c $USER) -eq 0 ]; then
        sudo usermod -aG docker $USER
        newgrp docker
    fi
    
    if [ -d "output" ]; then
        if [ "$(ls -ld "output" | awk '{print $3}')" = "root" ]; then
            sudo chown -R $RUN_UID:$RUN_UID .
        fi
    fi
}

function run_images() {
    SHDIR="$(cd "$(dirname $0)" && pwd)"
    CONTAINER_CMD="\"bash build.sh $ARGS\""
    echo -e "\033[32m"run images: $IMAGES_NAME."\033[0m"
    ENV=" -it -w "$SHDIR" -v "$SHDIR:$SHDIR" -e DISPLAY=unix$""DISPLAY --rm -u=${RUN_UID} --net=host "
    COMMAND="docker run"${ENV}${IMAGES_NAME}" bash -c $CONTAINER_CMD"
    echo $COMMAND
    sh -c "$COMMAND"
}

function main() {
    check_tools
    check_registries
    update_images
    run_images
}

main "$@"
