#ifndef _RKNN_CAR_RECOGNITION_H_
#define _RKNN_CAR_RECOGNITION_H_

#include "rknn_api.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define DETECT_MODEL_INPUT_SIZE 384
#define PLATE_MODEL_HEIGHT 48
#define PLATE_MODEL_WIDTH 168
#define CAR_MODEL_SIZE 64

#define OBJ_NUMB_MAX_SIZE 128
#define NMS_THRESH 0.5f
#define BOX_THRESH 0.3f

#define PLATE_CHARS_NUM 78
#define PLATE_SEQ_LEN 21
#define PLATE_COLOR_NUM 5
#define CAR_COLOR_NUM 11

#if defined(RV1106_1103)
#include "dma_alloc.hpp"
typedef struct {
    char *dma_buf_virt_addr;
    int dma_buf_fd;
    int size;
} rknn_dma_buf;
#endif

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
#if defined(RV1106_1103)
    rknn_tensor_mem *input_mems[1];
    rknn_tensor_mem *output_mems[1];
    rknn_dma_buf img_dma_buf;
#endif
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

typedef struct {
    float x1, y1, x2, y2;
    float conf;
    int cls_id;
} detect_box_t;

typedef struct {
    float x, y;
} landmark_point_t;

typedef struct {
    detect_box_t box;
    landmark_point_t landmarks[4];
    int class_id;
    float confidence;
} detect_result_t;

typedef struct {
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_list_t;

typedef struct {
    std::string plate_number;
    std::string plate_color;
    int color_id;
} plate_result_t;

typedef struct {
    std::string color_name;
    int color_id;
    float confidence;
} car_color_result_t;

typedef struct {
    int class_type;
    cv::Rect rect;
    std::vector<cv::Point2f> landmarks;
    std::string plate_number;
    std::string plate_color;
    std::string car_color;
    float confidence;
    float color_confidence;
} final_result_t;

typedef struct {
    rknn_app_context_t detect_ctx;
    rknn_app_context_t plate_ctx;
    rknn_app_context_t car_ctx;
    bool initialized;
} car_recognition_pipeline_t;

extern const std::vector<std::string> plate_chars;
extern const std::vector<std::string> plate_colors;
extern const std::vector<std::string> car_colors;
extern const std::vector<std::string> class_names;

int init_detect_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_detect_model(rknn_app_context_t *app_ctx);
int inference_detect_model(rknn_app_context_t *app_ctx, cv::Mat &img, detect_result_list_t *det_results, 
                           float conf_threshold, float nms_threshold);

int init_plate_rec_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_plate_rec_model(rknn_app_context_t *app_ctx);
int inference_plate_rec_model(rknn_app_context_t *app_ctx, cv::Mat &plate_img, plate_result_t *plate_result);

int init_car_color_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_car_color_model(rknn_app_context_t *app_ctx);
int inference_car_color_model(rknn_app_context_t *app_ctx, cv::Mat &car_img, car_color_result_t *color_result);

cv::Mat letterbox(cv::Mat &img, int new_shape, cv::Scalar color);
cv::Mat perspective_transform(cv::Mat &img, const std::vector<cv::Point2f> &points);
cv::Mat split_double_plate(cv::Mat &plate_img);

void draw_results(cv::Mat &img, const std::vector<final_result_t> &results);

int init_car_recognition_pipeline(car_recognition_pipeline_t *pipeline,
                                  const char *detect_model_path,
                                  const char *plate_model_path,
                                  const char *car_model_path);
int release_car_recognition_pipeline(car_recognition_pipeline_t *pipeline);
int inference_car_recognition_pipeline(car_recognition_pipeline_t *pipeline,
                                       cv::Mat &img,
                                       std::vector<final_result_t> *final_results,
                                       double *elapsed_ms);

#endif
