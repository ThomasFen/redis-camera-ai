# Analze emotions and see the result in real time

1. Pull docker images and start containers

```
docker-compose up
```

2. Load models, gear and script into redis
```
cd ai/
cat blazeface_back_converted.onnx | redis-cli -h redis -x AI.MODELSTORE blazeface:model ONNX CPU BLOB
cat emotion.onnx | redis-cli -h redis -x AI.MODELSTORE emotion ONNX CPU BLOB
cat blazeface-script.py | redis-cli -h redis -x AI.SCRIPTSET blazeface:script CPU SOURCE
cat gears-emotion.py | redis-cli -x RG.PYEXECUTE
```

3. Add camera frames to redis stream

```
cd camera/
python3 edge-camera.py -u redis://redis:6379
```
4. [Visit the dashboard](http://localhost:3000/d/DtsbVE3Mk/camera-processing?orgId=1) (Click refresh in the upper right corner to see frame latency updates)
