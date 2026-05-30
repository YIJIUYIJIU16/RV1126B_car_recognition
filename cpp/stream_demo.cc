#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <string.h>
#include <list>
#include <string>
#include <vector>

#include <linux/videodev2.h>

#include "car_recognition.h"
#include "V4l2Device.h"
#include "V4l2Capture.h"

int LogLevel = 0;

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -d, --detect_model  Detection model path (default: model/detect.rknn)\n");
    printf("  -p, --plate_model   Plate recognition model path (default: model/plate_rec_color.rknn)\n");
    printf("  -c, --car_model     Car color model path (default: model/car_recognition.rknn)\n");
    printf("  -D, --device        V4L2 device path (default: /dev/video0)\n");
    printf("  -W, --width         Capture width (default: 640)\n");
    printf("  -H, --height        Capture height (default: 480)\n");
    printf("  -f, --fps           Capture fps (default: 30)\n");
    printf("  -m, --max_frames    Stop after N processed frames (default: 0 means unlimited)\n");
    printf("  -h, --help          Show this help message\n");
}

int main(int argc, char **argv)
{
    const char *detect_model_path = "model/detect.rknn";
    const char *plate_model_path = "model/plate_rec_color.rknn";
    const char *car_model_path = "model/car_recognition.rknn";
    const char *device_path = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;
    int max_frames = 0;

    static struct option long_options[] = {
        {"detect_model", required_argument, 0, 'd'},
        {"plate_model", required_argument, 0, 'p'},
        {"car_model", required_argument, 0, 'c'},
        {"device", required_argument, 0, 'D'},
        {"width", required_argument, 0, 'W'},
        {"height", required_argument, 0, 'H'},
        {"fps", required_argument, 0, 'f'},
        {"max_frames", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:p:c:D:W:H:f:m:h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'd':
            detect_model_path = optarg;
            break;
        case 'p':
            plate_model_path = optarg;
            break;
        case 'c':
            car_model_path = optarg;
            break;
        case 'D':
            device_path = optarg;
            break;
        case 'W':
            width = atoi(optarg);
            break;
        case 'H':
            height = atoi(optarg);
            break;
        case 'f':
            fps = atoi(optarg);
            break;
        case 'm':
            max_frames = atoi(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return -1;
        }
    }

    car_recognition_pipeline_t pipeline;
    int ret = init_car_recognition_pipeline(&pipeline, detect_model_path, plate_model_path, car_model_path);
    if (ret != 0)
    {
        printf("init_car_recognition_pipeline fail! ret=%d\n", ret);
        return -1;
    }

    std::list<uint32_t> format_list;
    format_list.push_back(V4L2_PIX_FMT_YUYV);
    V4L2DeviceParameters param(device_path, format_list, width, height, fps, IOTYPE_MMAP, 0);
    V4l2Capture *capture = V4l2Capture::create(param, V4L2_BUF_TYPE_VIDEO_CAPTURE);
    if (capture == NULL)
    {
        printf("Failed to open capture device: %s\n", device_path);
        release_car_recognition_pipeline(&pipeline);
        return -1;
    }

    std::vector<unsigned char> buffer(capture->getBufferSize());

    printf("=== Car Recognition Stream Demo ===\n");
    printf("Device: %s\n", device_path);
    printf("Capture: %dx%d @ %d fps\n", width, height, fps);

    int frame_index = 0;
    while (true)
    {
        timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;

        int readable = capture->isReadable(&tv);
        if (readable == -1)
        {
            printf("Capture stopped: %s\n", strerror(errno));
            break;
        }
        if (readable != 1)
        {
            continue;
        }

        int read_size = static_cast<int>(capture->read(reinterpret_cast<char *>(buffer.data()), buffer.size()));
        if (read_size <= 0)
        {
            printf("Frame read failed: %s\n", strerror(errno));
            break;
        }

        cv::Mat yuyv(height, width, CV_8UC2, buffer.data());
        cv::Mat frame;
        cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUY2);

        std::vector<final_result_t> final_results;
        double elapsed_ms = 0.0;
        ret = inference_car_recognition_pipeline(&pipeline, frame, &final_results, &elapsed_ms);
        if (ret != 0)
        {
            printf("Frame %d inference failed! ret=%d\n", frame_index, ret);
            break;
        }

        draw_results(frame, final_results);
        printf("Frame %d: objects=%zu, time=%.2f ms, fps=%.2f\n",
               frame_index, final_results.size(), elapsed_ms, elapsed_ms > 0.0 ? 1000.0 / elapsed_ms : 0.0);

        frame_index++;
        if (max_frames > 0 && frame_index >= max_frames)
        {
            printf("Reached max_frames=%d\n", max_frames);
            break;
        }
    }

    delete capture;
    release_car_recognition_pipeline(&pipeline);
    return 0;
}
