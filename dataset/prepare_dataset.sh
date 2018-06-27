#!/bin/bash
set -e

cur_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p ${cur_dir}/data
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O "${cur_dir}/data/cifar-10-python.tar.gz"
cd ${cur_dir}/data && tar -zxvf cifar-10-python.tar.gz