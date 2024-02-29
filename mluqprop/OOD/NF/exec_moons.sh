if [ ! -f ../data/moons.npy ]
then
    echo "Generating Moons"
    cd ../data
    python makeMoons.py
    cd ../NF
fi
python main.py -i input_moons
python plotResult.py -i input_moons
python plotLoss.py -i input_moons
