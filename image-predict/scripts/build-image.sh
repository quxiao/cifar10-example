#!/bin/bash

cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $cur_dir/../..
push=false

if [ "$1" == "--push" ];then
    push=true
fi

docker build -t cifar-image-predict:latest -f $cur_dir/Dockerfile .
if $push; then
    docker tag cifar-image-predict:latest reg.qiniu.com/quxiao-public/cifar-image-predict:latest
    docker push reg.qiniu.com/quxiao-public/cifar-image-predict:latest
fi