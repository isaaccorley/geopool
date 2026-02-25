mkdir data
cd data
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-aef.tar.gz
tar -xvzf eurosat-aef.tar.gz
rm eurosat-aef.tar.gz
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-spatial-train.txt
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-spatial-val.txt
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-spatial-test.txt
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-train.txt
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-val.txt
wget https://hf.co/datasets/isaaccorley/AlphaEarth-EuroSAT/resolve/main/eurosat-test.txt
cd ..