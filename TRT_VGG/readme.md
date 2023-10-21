# Prepare Wegihts
```bash
cd weight
python vgg.py
```

# Compile
```bash
mkdir build
cd build
cmake ..
make
# serialize
./vgg_demo -s
# deserialize
./vgg_demo -d
```

# Test by ljf
<div align=center>
<img src ="./python.png" width="800"/>
<img src ="./cpp.png" width="800"/>
</div>


