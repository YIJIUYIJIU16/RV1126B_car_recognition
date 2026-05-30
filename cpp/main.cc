#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <getopt.h>
#include <algorithm>
#include <dirent.h>
#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "car_recognition.h"
#include "image_utils.h"
#include "file_utils.h"

static const char *class_names_cn[] = {"单层车牌", "双层车牌", "汽车"};
static const char *plate_colors_cn[] = {"黑色", "蓝色", "绿色", "白色", "黄色"};
static const char *car_colors_cn[] = {"黑色", "蓝色", "黄色", "棕色", "绿色", "灰色", "橙色", "粉色", "紫色", "红色", "白色"};

static bool has_image_extension(const std::string &path)
{
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos)
    {
        return false;
    }

    std::string ext = path.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp";
}

static bool load_image_with_fallback(const std::string &path, cv::Mat &img)
{
    img = cv::imread(path);
    if (!img.empty())
    {
        return true;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    if (read_image(path.c_str(), &src_image) != 0 || src_image.virt_addr == NULL)
    {
        return false;
    }

    if (src_image.format == IMAGE_FORMAT_RGB888)
    {
        cv::Mat rgb(src_image.height, src_image.width, CV_8UC3, src_image.virt_addr);
        cv::cvtColor(rgb, img, cv::COLOR_RGB2BGR);
    }
    else if (src_image.format == IMAGE_FORMAT_RGBA8888)
    {
        cv::Mat rgba(src_image.height, src_image.width, CV_8UC4, src_image.virt_addr);
        cv::cvtColor(rgba, img, cv::COLOR_RGBA2BGR);
    }
    else if (src_image.format == IMAGE_FORMAT_GRAY8)
    {
        cv::Mat gray(src_image.height, src_image.width, CV_8UC1, src_image.virt_addr);
        cv::cvtColor(gray, img, cv::COLOR_GRAY2BGR);
    }

    free(src_image.virt_addr);
    return !img.empty();
}

static bool path_exists(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static bool is_directory_path(const std::string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static std::string join_path(const std::string &base, const std::string &name)
{
    if (base.empty())
    {
        return name;
    }
    if (base.back() == '/')
    {
        return base + name;
    }
    return base + "/" + name;
}

static std::string get_basename(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return path;
    }
    return path.substr(pos + 1);
}

static std::string get_parent_dir(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
    {
        return "";
    }
    return path.substr(0, pos);
}

static bool ensure_directory_recursive(const std::string &dir_path)
{
    if (dir_path.empty() || dir_path == ".")
    {
        return true;
    }
    if (is_directory_path(dir_path))
    {
        return true;
    }

    std::string normalized = dir_path;
    while (normalized.size() > 1 && normalized.back() == '/')
    {
        normalized.pop_back();
    }

    size_t pos = normalized.find_last_of('/');
    if (pos != std::string::npos)
    {
        std::string parent = normalized.substr(0, pos);
        if (!parent.empty() && !ensure_directory_recursive(parent))
        {
            return false;
        }
    }

    if (mkdir(normalized.c_str(), 0755) == 0 || errno == EEXIST)
    {
        return true;
    }

    printf("Failed to create directory: %s, errno=%d\n", normalized.c_str(), errno);
    return false;
}

static void collect_image_files_recursive(const std::string &root_dir, const std::string &relative_dir,
                                          std::vector<std::pair<std::string, std::string>> &files)
{
    std::string current_dir = relative_dir.empty() ? root_dir : join_path(root_dir, relative_dir);
    DIR *dir = opendir(current_dir.c_str());
    if (dir == NULL)
    {
        printf("Failed to open directory: %s\n", current_dir.c_str());
        return;
    }

    std::vector<std::string> entries;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        std::string name = entry->d_name;
        if (name == "." || name == "..")
        {
            continue;
        }
        entries.push_back(name);
    }
    closedir(dir);

    std::sort(entries.begin(), entries.end());
    for (const auto &name : entries)
    {
        std::string rel_path = relative_dir.empty() ? name : join_path(relative_dir, name);
        std::string full_path = join_path(root_dir, rel_path);
        if (is_directory_path(full_path))
        {
            collect_image_files_recursive(root_dir, rel_path, files);
        }
        else if (has_image_extension(full_path))
        {
            files.push_back(std::make_pair(full_path, rel_path));
        }
    }
}

static const char *to_plate_color_cn(const std::string &name)
{
    for (size_t i = 0; i < plate_colors.size() && i < sizeof(plate_colors_cn) / sizeof(plate_colors_cn[0]); ++i)
    {
        if (plate_colors[i] == name)
        {
            return plate_colors_cn[i];
        }
    }
    return "";
}

static const char *to_car_color_cn(const std::string &name)
{
    for (size_t i = 0; i < car_colors.size() && i < sizeof(car_colors_cn) / sizeof(car_colors_cn[0]); ++i)
    {
        if (car_colors[i] == name)
        {
            return car_colors_cn[i];
        }
    }
    return "";
}

static void print_usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -d, --detect_model  Detection model path (default: model/detect.rknn)\n");
    printf("  -p, --plate_model   Plate recognition model path (default: model/plate_rec_color.rknn)\n");
    printf("  -c, --car_model     Car color model path (default: model/car_recognition.rknn)\n");
    printf("  -i, --image         Input image path or directory path\n");
    printf("  -o, --output        Output image path or result directory (default: result.jpg)\n");
    printf("  -h, --help          Show this help message\n");
}

static int process_single_image(rknn_app_context_t *detect_ctx, rknn_app_context_t *plate_ctx,
                                rknn_app_context_t *car_ctx, const std::string &image_path,
                                const std::string &output_path)
{
    int ret;

    printf("Loading image from: %s\n", image_path.c_str());
    cv::Mat img;
    bool image_ok = load_image_with_fallback(image_path, img);
    if (!image_ok || img.empty())
    {
        printf("Failed to read image: %s\n", image_path.c_str());
        return -1;
    }
    printf("Image loaded successfully, size: %dx%d, channels: %d\n", img.cols, img.rows, img.channels());

    auto start = std::chrono::high_resolution_clock::now();

    printf("Starting detection inference...\n");
    detect_result_list_t det_results;
    ret = inference_detect_model(detect_ctx, img, &det_results, BOX_THRESH, NMS_THRESH);
    if (ret != 0)
    {
        printf("Detection inference failed! ret=%d\n", ret);
        return -1;
    }
    printf("Detection inference completed, found %d objects\n", det_results.count);

    std::vector<final_result_t> final_results;
    for (int i = 0; i < det_results.count; i++)
    {
        printf("Processing object %d/%d\n", i + 1, det_results.count);
        detect_result_t *det = &det_results.results[i];
        final_result_t final_res;
        final_res.class_type = det->class_id;
        final_res.rect = cv::Rect(det->box.x1, det->box.y1,
                                  det->box.x2 - det->box.x1,
                                  det->box.y2 - det->box.y1);
        final_res.confidence = det->confidence;

        for (int j = 0; j < 4; j++)
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
                    ret = inference_car_color_model(car_ctx, car_img, &color_res);
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
                std::vector<cv::Point2f> pts;
                for (const auto &pt : final_res.landmarks)
                {
                    pts.push_back(pt);
                }

                cv::Mat warped = perspective_transform(img, pts);
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
                    ret = inference_plate_rec_model(plate_ctx, warped, &plate_res);
                    if (ret == 0)
                    {
                        final_res.plate_number = plate_res.plate_number;
                        final_res.plate_color = plate_res.plate_color;
                    }
                }
            }
        }

        final_results.push_back(final_res);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Drawing results...\n");
    fflush(stdout);
    draw_results(img, final_results);
    printf("Drawing completed\n");
    fflush(stdout);

    std::string parent_dir = get_parent_dir(output_path);
    if (!parent_dir.empty() && !ensure_directory_recursive(parent_dir))
    {
        printf("Failed to prepare output directory for: %s\n", output_path.c_str());
        return -1;
    }

    printf("Writing output to: %s\n", output_path.c_str());
    fflush(stdout);

    image_buffer_t dst_image;
    memset(&dst_image, 0, sizeof(image_buffer_t));
    dst_image.width = img.cols;
    dst_image.height = img.rows;
    dst_image.format = IMAGE_FORMAT_RGB888;
    dst_image.size = img.cols * img.rows * 3;
    dst_image.virt_addr = (unsigned char *)malloc(dst_image.size);
    if (dst_image.virt_addr == NULL)
    {
        printf("Failed to allocate memory for output image\n");
        return -1;
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    memcpy(dst_image.virt_addr, img_rgb.data, dst_image.size);

    ret = write_image(output_path.c_str(), &dst_image);
    if (ret == 0)
    {
        printf("Image saved successfully\n");
    }
    else
    {
        printf("Failed to save image, ret=%d\n", ret);
        free(dst_image.virt_addr);
        return -1;
    }
    free(dst_image.virt_addr);
    fflush(stdout);

    printf("\nDetection results:\n");
    printf("----------------\n");
    for (const auto &res : final_results)
    {
        int class_idx = std::min(res.class_type, static_cast<int>(class_names.size() - 1));
        int class_cn_idx = std::min(res.class_type, static_cast<int>(sizeof(class_names_cn) / sizeof(class_names_cn[0]) - 1));
        printf("Class: %s (%s)\n", class_names[class_idx].c_str(), class_names_cn[class_cn_idx]);
        printf("  Rect: [%d, %d, %d, %d]\n", res.rect.x, res.rect.y, res.rect.width, res.rect.height);
        printf("  Confidence: %.3f\n", res.confidence);

        if (res.class_type == 2)
        {
            printf("  Car color: %s (%s) (%.3f)\n", res.car_color.c_str(), to_car_color_cn(res.car_color), res.color_confidence);
        }
        else
        {
            printf("  Plate number: %s\n", res.plate_number.c_str());
            printf("  Plate color: %s (%s)\n", res.plate_color.c_str(), to_plate_color_cn(res.plate_color));
        }
        printf("\n");
    }

    printf("Inference time: %.2f ms\n", elapsed);
    printf("FPS: %.1f\n", 1000.0 / elapsed);
    printf("Result saved to: %s\n", output_path.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    const char *detect_model_path = "model/detect.rknn";
    const char *plate_model_path = "model/plate_rec_color.rknn";
    const char *car_model_path = "model/car_recognition.rknn";
    const char *image_path = NULL;
    const char *output_path = "result.jpg";

    static struct option long_options[] = {
        {"detect_model", required_argument, 0, 'd'},
        {"plate_model", required_argument, 0, 'p'},
        {"car_model", required_argument, 0, 'c'},
        {"image", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    printf("The Version: 1.0.0\n");
    fflush(stdout);
    while ((opt = getopt_long(argc, argv, "d:p:c:i:o:h", long_options, NULL)) != -1)
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
        case 'i':
            image_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return -1;
        }
    }

    if (image_path == NULL)
    {
        printf("Error: Input image path is required\n");
        print_usage(argv[0]);
        return -1;
    }

    int ret;
    rknn_app_context_t detect_ctx;
    rknn_app_context_t plate_ctx;
    rknn_app_context_t car_ctx;

    memset(&detect_ctx, 0, sizeof(rknn_app_context_t));
    memset(&plate_ctx, 0, sizeof(rknn_app_context_t));
    memset(&car_ctx, 0, sizeof(rknn_app_context_t));

    printf("=== Car Recognition Demo ===\n");
    printf("Detect model: %s\n", detect_model_path);
    printf("Plate model: %s\n", plate_model_path);
    printf("Car model: %s\n", car_model_path);

    ret = init_detect_model(detect_model_path, &detect_ctx);
    if (ret != 0)
    {
        printf("init_detect_model fail! ret=%d\n", ret);
        return -1;
    }

    ret = init_plate_rec_model(plate_model_path, &plate_ctx);
    if (ret != 0)
    {
        printf("init_plate_rec_model fail! ret=%d\n", ret);
        release_detect_model(&detect_ctx);
        return -1;
    }

    ret = init_car_color_model(car_model_path, &car_ctx);
    if (ret != 0)
    {
        printf("init_car_color_model fail! ret=%d\n", ret);
        release_detect_model(&detect_ctx);
        release_plate_rec_model(&plate_ctx);
        return -1;
    }

    printf("All models loaded successfully!\n\n");
    int final_ret = 0;
    std::string input_path_str = image_path;
    std::string output_path_str = output_path;
    if (is_directory_path(input_path_str))
    {
        std::vector<std::pair<std::string, std::string>> image_files;
        collect_image_files_recursive(input_path_str, "", image_files);
        if (image_files.empty())
        {
            printf("No images found in directory: %s\n", input_path_str.c_str());
            final_ret = -1;
        }
        else
        {
            std::string output_dir = output_path_str;
            if (output_dir == "result.jpg")
            {
                output_dir = "result";
            }
            else if (path_exists(output_dir) && !is_directory_path(output_dir))
            {
                printf("Output path exists and is not a directory: %s\n", output_dir.c_str());
                final_ret = -1;
            }
            else if (!path_exists(output_dir) && has_image_extension(output_dir))
            {
                printf("When input is a directory, output must be a directory path: %s\n", output_dir.c_str());
                final_ret = -1;
            }

            if (final_ret == 0 && !ensure_directory_recursive(output_dir))
            {
                final_ret = -1;
            }

            int success_count = 0;
            int fail_count = 0;
            for (size_t i = 0; final_ret == 0 && i < image_files.size(); ++i)
            {
                const std::string &full_path = image_files[i].first;
                const std::string &rel_path = image_files[i].second;
                std::string save_path = join_path(output_dir, rel_path);

                printf("\n=== Batch %zu/%zu ===\n", i + 1, image_files.size());
                if (process_single_image(&detect_ctx, &plate_ctx, &car_ctx, full_path, save_path) == 0)
                {
                    success_count++;
                }
                else
                {
                    fail_count++;
                }
            }

            printf("\n=== Batch Summary ===\n");
            printf("Input directory: %s\n", input_path_str.c_str());
            printf("Output directory: %s\n", output_dir.c_str());
            printf("Total images: %zu\n", image_files.size());
            printf("Success: %d\n", success_count);
            printf("Failed: %d\n", fail_count);
            if (success_count == 0)
            {
                final_ret = -1;
            }
        }
    }
    else
    {
        std::string final_output_path = output_path_str;
        if (is_directory_path(final_output_path))
        {
            final_output_path = join_path(final_output_path, get_basename(input_path_str));
        }
        final_ret = process_single_image(&detect_ctx, &plate_ctx, &car_ctx, input_path_str, final_output_path);
    }

    release_detect_model(&detect_ctx);
    release_plate_rec_model(&plate_ctx);
    release_car_color_model(&car_ctx);

    return final_ret;
}
