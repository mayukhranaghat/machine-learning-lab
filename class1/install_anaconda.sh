#!/bin/bash
sudo apt-get update
sudo apt-get upgrade
wget -P ~/Downloads https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i ~/Downloads/google-chrome-stable_current_amd64.deb
sudo add-apt-repository ppa:notepadqq-team/notepadqq
sudo apt-get update
sudo apt-get install notepadqq-gtk
sudo apt-get upgrade
sudo gsettings set com.canonical.Unity.Launcher favorites "['unity://running-apps', 'application://notepadqq.desktop', 'application://org.gnome.Nautilus.desktop', 'application://google-chrome.desktop', 'unity://expo-icon', 'unity://devices']"
sudo apt-get install default-jre
sudo apt-get install default-jdk
wget -P ~/Downloads https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash ~/Downloads/Anaconda3-2019.03-Linux-x86_64.sh



