#!/bin/bash
set -e
echo "0 - Replica sample (one scene)"
echo "1 - Replca"
echo "2 - ScanNet"
read -p "Enter the dataset ID you want to download: " ds_id

mkdir -p data
cd data

if [ $ds_id == 0 ]
then
    echo "Download Replica sample dataset, room00"
    # Download Replica scenes used in objectsdf++, includine rgb, depth, normal, semantic, instance, pose
    gdown --no-cookies 17U8RzDWCtUCNPTDF16pEhFqznbz5u8bc -O replica_sample.zip
    echo "done,start unzipping"
    unzip -o replica_sample.zip -d replica
    rm -rf replica_sample.zip

elif [ $ds_id == 1 ]
then
    echo "Download All Replica dataset, 8 scenes"
    # Download Replica scenes used in objectsdf++, includine rgb, depth, normal, semantic, instance, pose
    gdown --no-cookies 1IAFNQE3TNyE_ZNdJhDCcPPebWqbTuzYl -O replica.zip
    echo "done,start unzipping"
    unzip -o replica.zip
    rm -rf replica.zip

elif [ $ds_id == 2 ]
then
    # Download scannet scenes follow monosdf
    echo "Download ScanNet dataset, 4 scenes"
    gdown --no-cookies 1w-HZHhhvc71xOYhFBdZrLYu8FBsNWBhU -O scannet.zip
    echo "done,start unzipping"
    unzip -o scannet.zip
    rm -rf scannet.zip

else
    echo "Invalid dataset ID"
fi

cd ..