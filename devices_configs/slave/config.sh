#!/bin/bash

# Update package index
sudo apt-get update

# Install Docker and Docker Compose
sudo apt-get install -y docker docker-compose

cmdline="/boot/cmdline.txt"
new_options="cgroup_enable=cpu_set cgroup_enable=memory cgroup_memory=1 swapaccount=1"

sudo sed -i "s/\(.*\)/\1 $new_options/" "$cmd_line"

# Create docker group and add current user
sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker