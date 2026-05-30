#include <algorithm>
#include <chrono>
#include <string.h>
#include <vector>

#include "car_recognition.h"

int init_car_recognition_pipeline(car_recognition_pipeline_t *pipeline,
                                  const char *detect_model_path,
                                  const char *plate_model_path,
                                  const char *car_model_path)
{
    if (pipeline == NULL)
    {
        return -1;
    }

    memset(pipeline, 0, sizeof(*pipeline));

    int ret = init_detect_model(detect_model_path, &pipeline->detect_ctx);
    if (ret != 0)
    {
        return ret;
    }

    ret = init_plate_rec_model(plate_model_path, &pipeline->plate_ctx);
    if (ret != 0)
    {
        release_detect_model(&pipeline->detect_ctx);
        memset(&pipeline->detect_ctx, 0, sizeof(pipeline->detect_ctx));
        return ret;
    }

    ret = init_car_color_model(car_model_path, &pipeline->car_ctx);
    if (ret != 0)
    {
        release_detect_model(&pipeline->detect_ctx);
        release_plate_rec_model(&pipeline->plate_ctx);
        memset(&pipeline->detect_ctx, 0, sizeof(pipeline->detect_ctx));
        memset(&pipeline->plate_ctx, 0, sizeof(pipeline->plate_ctx));
        return ret;
    }

    pipeline->initialized = true;
    return 0;
}

int release_car_recognition_pipeline(car_recognition_pipeline_t *pipeline)
{
    if (pipeline == NULL)
    {
        return -1;
    }

    release_detect_model(&pipeline->detect_ctx);
    release_plate_rec_model(&pipeline->plate_ctx);
    release_car_color_model(&pipeline->car_ctx);
    memset(pipeline, 0, sizeof(*pipeline));
    return 0;
}

int inference_car_recognition_pipeline(car_recognition_pipeline_t *pipeline,
                                       cv::Mat &img,
                                       std::vector<final_result_t> *final_results,
                                       double *elapsed_ms)
{
    if (pipeline == NULL || final_results == NULL || !pipeline->initialized || img.empty())
    {
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();

    detect_result_list_t det_results;
    int ret = inference_detect_model(&pipeline->detect_ctx, img, &det_results, BOX_THRESH, NMS_THRESH);
    if (ret != 0)
    {
        return ret;
    }

    final_results->clear();
    final_results->reserve(det_results.count);

    for (int i = 0; i < det_results.count; ++i)
    {
        detect_result_t *det = &det_results.results[i];
        final_result_t final_res;
        final_res.class_type = det->class_id;
        final_res.rect = cv::Rect(det->box.x1, det->box.y1,
                                  det->box.x2 - det->box.x1,
                                  det->box.y2 - det->box.y1);
        final_res.confidence = det->confidence;
        final_res.color_confidence = 0.0f;

        for (int j = 0; j < 4; ++j)
        {
            final_res.landmarks.push_back(cv::Point2f(det->landmarks[j].x, det->landmarks[j].y));
        }

        if (det->class_id == 2)
        {
            cv::Rect roi = final_res.rect & cv::Rect(0, 0, img.cols, img.rows);
            if (roi.area() > 0)
            {
                cv::Mat car_img = img(roi);
                if (!car_img.empty())
                {
                    car_color_result_t color_res;
                    ret = inference_car_color_model(&pipeline->car_ctx, car_img, &color_res);
                    if (ret == 0)
                    {
                        final_res.car_color = color_res.color_name;
                        final_res.color_confidence = color_res.confidence;
                    }
                }
            }
        }
        else
        {
            cv::Rect roi = final_res.rect & cv::Rect(0, 0, img.cols, img.rows);
            if (roi.area() > 0 && final_res.landmarks.size() == 4)
            {
                cv::Mat warped = perspective_transform(img, final_res.landmarks);
                if (!warped.empty())
                {
                    if (det->class_id == 1)
                    {
                        cv::Mat split_result = split_double_plate(warped);
                        if (!split_result.empty())
                        {
                            warped = split_result;
                        }
                    }

                    plate_result_t plate_res;
                    ret = inference_plate_rec_model(&pipeline->plate_ctx, warped, &plate_res);
                    if (ret == 0)
                    {
                        final_res.plate_number = plate_res.plate_number;
                        final_res.plate_color = plate_res.plate_color;
                    }
                }
            }
        }

        final_results->push_back(final_res);
    }

    auto end = std::chrono::high_resolution_clock::now();
    if (elapsed_ms != NULL)
    {
        *elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return 0;
}
