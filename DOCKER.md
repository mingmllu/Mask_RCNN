### Dockerize the mask RCNN for serving

```
docker build --no-cache --build-arg http_proxy=$http_proxy --build-arg https_proxy=$http_proxy -t mmlu_test_py36k:v0.1 .
```

### Use the Docker image

```
$ docker run -it --rm -v ~/Workspace:/mask_rcnn/videos/ mmlu_test_py36k:v0.1
```
If you want to read external video source (e.g., http://108.53.114.166/mjpg/video.mjpg) from behind corporate network, you need to use provide proxy server:
```
$ docker run -e http_proxy=$http_proxy -e https_proxy=$https_proxy --rm -v ~/Workspace:/mask_rcnn/videos/ mmlu_test_py36k:v0.1
```
