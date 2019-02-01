### Dockerize the mask RCNN for serving

```
docker build --no-cache --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy -t mmlu/mask_rcnn_gpu:v0.1 .
```

### Use the Docker image

```
$ docker run -it --rm -v ~/Workspace:/mask_rcnn/videos/ mmlu_test_py36k:v0.1
```
If you want to read external video source (e.g., http://108.53.114.166/mjpg/video.mjpg) from behind corporate network, you need to use provide proxy server:
```
$ docker run -e http_proxy=$http_proxy -e https_proxy=$https_proxy --rm -v ~/Workspace:/mask_rcnn/videos/ mmlu_test_py36k:v0.1
```
### Run Docker images on GPU machine and see the processed images on another machine

Dockerfile:
```
[...]
CMD ["python3.5", "segmt-live-video-zmq.py"]
```
Build image:
```
docker build --no-cache --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy -t mmlu_test_0mq:v0.1 .
```
Run "video_zmq_client.py" on another machine. First change the line: ```SERVICE_SOCKET='tcp://localhost:5566'``` using the GPU machine IP address:
```
SERVICE_SOCKET='tcp://135.222.154.125:5566'
```
Then run the client:
```
$ python video_zmq_client.py
```
Of course, you have to set up a virtual environment and install required packages.

Run the image on GPU workstation:
```
docker run -it -e SKT_PORT=5566 --expose=5566 -p 5566:5566 --rm -v ~/Workspace:/mask_rcnn/videos/ mmlu_test_0mq:v0.1
```

