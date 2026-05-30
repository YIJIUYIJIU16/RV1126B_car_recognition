#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <sstream>
#include <vector>

#include "car_recognition.h"

static const char *kPlateGlyphDir = "model/platech_glyphs";
static const int kGlyphBaseFontSize = 64;

static const char *to_plate_color_cn_draw(const std::string &name)
{
    if (name == "Black") return "黑色";
    if (name == "Blue") return "蓝色";
    if (name == "Green") return "绿色";
    if (name == "White") return "白色";
    if (name == "Yellow") return "黄色";
    return name.c_str();
}

static const char *to_car_color_cn_draw(const std::string &name)
{
    if (name == "Black") return "黑色";
    if (name == "Blue") return "蓝色";
    if (name == "Yellow") return "黄色";
    if (name == "Brown") return "棕色";
    if (name == "Green") return "绿色";
    if (name == "Gray") return "灰色";
    if (name == "Orange") return "橙色";
    if (name == "Pink") return "粉色";
    if (name == "Purple") return "紫色";
    if (name == "Red") return "红色";
    if (name == "White") return "白色";
    return name.c_str();
}

static bool decode_utf8_char(const std::string &text, size_t start, uint32_t *codepoint, size_t *length)
{
    if (start >= text.size())
    {
        return false;
    }

    const unsigned char *s = reinterpret_cast<const unsigned char *>(text.data());
    unsigned char c = s[start];
    if (c < 0x80)
    {
        *codepoint = c;
        *length = 1;
        return true;
    }
    if ((c >> 5) == 0x6 && start + 1 < text.size())
    {
        *codepoint = ((c & 0x1F) << 6) | (s[start + 1] & 0x3F);
        *length = 2;
        return true;
    }
    if ((c >> 4) == 0xE && start + 2 < text.size())
    {
        *codepoint = ((c & 0x0F) << 12) | ((s[start + 1] & 0x3F) << 6) | (s[start + 2] & 0x3F);
        *length = 3;
        return true;
    }
    if ((c >> 3) == 0x1E && start + 3 < text.size())
    {
        *codepoint = ((c & 0x07) << 18) | ((s[start + 1] & 0x3F) << 12) |
                     ((s[start + 2] & 0x3F) << 6) | (s[start + 3] & 0x3F);
        *length = 4;
        return true;
    }
    return false;
}

static std::string glyph_path_for_codepoint(uint32_t codepoint)
{
    std::ostringstream oss;
    oss << kPlateGlyphDir << "/u";
    oss << std::uppercase << std::hex;
    oss.width(4);
    oss.fill('0');
    oss << codepoint << ".png";
    return oss.str();
}

static cv::Mat load_glyph_rgba(uint32_t codepoint)
{
    static std::map<uint32_t, cv::Mat> glyph_cache;
    auto it = glyph_cache.find(codepoint);
    if (it != glyph_cache.end())
    {
        return it->second;
    }

    cv::Mat glyph = cv::imread(glyph_path_for_codepoint(codepoint), cv::IMREAD_UNCHANGED);
    glyph_cache[codepoint] = glyph;
    return glyph;
}

static void alpha_blend_rgba(cv::Mat &dst, const cv::Mat &src_rgba, int x, int y, const cv::Scalar &color)
{
    if (src_rgba.empty() || src_rgba.channels() != 4)
    {
        return;
    }

    for (int row = 0; row < src_rgba.rows; ++row)
    {
        int dst_y = y + row;
        if (dst_y < 0 || dst_y >= dst.rows)
        {
            continue;
        }

        const cv::Vec4b *src_ptr = src_rgba.ptr<cv::Vec4b>(row);
        cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(dst_y);
        for (int col = 0; col < src_rgba.cols; ++col)
        {
            int dst_x = x + col;
            if (dst_x < 0 || dst_x >= dst.cols)
            {
                continue;
            }

            float alpha = src_ptr[col][3] / 255.0f;
            if (alpha <= 0.0f)
            {
                continue;
            }

            for (int c = 0; c < 3; ++c)
            {
                float fg = static_cast<float>(color[c]);
                float bg = static_cast<float>(dst_ptr[dst_x][c]);
                dst_ptr[dst_x][c] = static_cast<unsigned char>(bg * (1.0f - alpha) + fg * alpha);
            }
        }
    }
}

static int draw_utf8_text(cv::Mat &img, const std::string &text, cv::Point origin,
                          const cv::Scalar &color, int target_height)
{
    if (img.empty() || text.empty())
    {
        return 0;
    }

    int cursor_x = origin.x;
    int top_y = origin.y;
    float scale = std::max(8, target_height) / static_cast<float>(kGlyphBaseFontSize);
    bool drew_any = false;

    for (size_t i = 0; i < text.size();)
    {
        uint32_t codepoint = 0;
        size_t char_len = 0;
        if (!decode_utf8_char(text, i, &codepoint, &char_len))
        {
            i++;
            continue;
        }

        if (codepoint == ' ')
        {
            cursor_x += std::max(4, static_cast<int>(std::round(target_height * 0.35f)));
            i += char_len;
            continue;
        }

        cv::Mat glyph = load_glyph_rgba(codepoint);
        if (glyph.empty())
        {
            std::string fallback = text.substr(i, char_len);
            cv::putText(img, fallback, cv::Point(cursor_x, top_y + std::max(12, target_height)),
                        cv::FONT_HERSHEY_SIMPLEX, target_height / 24.0, color, 1);
            cursor_x += std::max(8, target_height / 2);
            i += char_len;
            continue;
        }

        int draw_w = std::max(1, static_cast<int>(std::round(glyph.cols * scale)));
        int draw_h = std::max(1, static_cast<int>(std::round(glyph.rows * scale)));
        cv::Mat glyph_resized;
        cv::resize(glyph, glyph_resized, cv::Size(draw_w, draw_h), 0, 0, cv::INTER_LINEAR);
        alpha_blend_rgba(img, glyph_resized, cursor_x, top_y, color);
        cursor_x += draw_w + std::max(1, static_cast<int>(std::round(target_height * 0.06f)));
        drew_any = true;
        i += char_len;
    }

    return drew_any ? (cursor_x - origin.x) : 0;
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static std::vector<cv::Point2f> order_points(const std::vector<cv::Point2f> &points)
{
    if (points.size() != 4)
    {
        return points;
    }

    std::vector<cv::Point2f> rect(4);
    std::vector<float> sums(4), diffs(4);
    for (size_t i = 0; i < points.size(); ++i)
    {
        sums[i] = points[i].x + points[i].y;
        diffs[i] = points[i].y - points[i].x;
    }

    rect[0] = points[std::min_element(sums.begin(), sums.end()) - sums.begin()];
    rect[2] = points[std::max_element(sums.begin(), sums.end()) - sums.begin()];
    rect[1] = points[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()];
    rect[3] = points[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()];
    return rect;
}

// 78 characters total: 1 blank + 31 provinces + 10 special + 10 digits + 24 letters + 2 special chars
const std::vector<std::string> plate_chars = {
    // Index 0: blank/unknown
    "#",
    // Index 1-31: Province abbreviations (31 provinces)
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    // Index 32-41: Special characters (10)
    "学", "警", "港", "澳", "挂", "使", "领", "民", "航", "危",
    // Index 42-51: Digits (10)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    // Index 52-75: Letters (24, excluding I and O)
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    // Index 76-77: Additional special characters (2)
    "险", "品"
};

const std::vector<std::string> plate_colors = {
    "Black", "Blue", "Green", "White", "Yellow"
};

const std::vector<std::string> car_colors = {
    "Black", "Blue", "Yellow", "Brown", "Green", "Gray", "Orange", "Pink", "Purple", "Red", "White"
};

const std::vector<std::string> class_names = {
    "SinglePlate", "DoublePlate", "Car"
};

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }
    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%d, type=%d, qnt_type=%d, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size,
           attr->fmt, attr->type, attr->qnt_type, attr->zp, attr->scale);
}

static char *read_model_data(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);
    char *data = (char *)malloc(size);
    if (data == nullptr)
    {
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);
    if (fread(data, 1, size, fp) != size)
    {
        free(data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *model_size = size;
    return data;
}

static float iou(const detect_box_t &a, const detect_box_t &b)
{
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;
    
    return intersection / (union_area + 1e-6f);
}

static std::vector<int> nms(std::vector<detect_box_t> &boxes, float iou_threshold)
{
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < boxes.size(); i++)
    {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(), [&boxes](int i, int j) {
        return boxes[i].conf > boxes[j].conf;
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (int i : indices)
    {
        if (suppressed[i])
            continue;
        keep.push_back(i);
        
        for (int j : indices)
        {
            if (suppressed[j])
                continue;
            if (boxes[i].cls_id == boxes[j].cls_id && iou(boxes[i], boxes[j]) > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }
    
    return keep;
}

int init_detect_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    model = read_model_data(model_path, &model_len);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("detect model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("detect input tensors:\n");
    rknn_tensor_attr *input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    if (input_attrs == NULL) {
        printf("malloc input_attrs failed!\n");
        return -1;
    }
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            free(input_attrs);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("detect output tensors:\n");
    rknn_tensor_attr *output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (output_attrs == NULL) {
        printf("malloc output_attrs failed!\n");
        free(input_attrs);
        return -1;
    }
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            free(input_attrs);
            free(output_attrs);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        app_ctx->is_quant = true;
    }
    else
    {
        app_ctx->is_quant = false;
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("detect model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("detect model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("detect model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    free(input_attrs);
    free(output_attrs);
    return 0;
}

int release_detect_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_detect_model(rknn_app_context_t *app_ctx, cv::Mat &img, detect_result_list_t *det_results,
                           float conf_threshold, float nms_threshold)
{
    printf("inference_detect_model: enter\n");
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));
    memset(det_results, 0, sizeof(detect_result_list_t));

    if (img.empty())
    {
        printf("Input image is empty!\n");
        return -1;
    }

    printf("inference_detect_model: calling letterbox\n");
    cv::Mat letterboxed = letterbox(img, app_ctx->model_width, cv::Scalar(114, 114, 114));
    if (letterboxed.empty())
    {
        printf("Letterbox failed!\n");
        return -1;
    }
    printf("inference_detect_model: letterbox done, size: %dx%d\n", letterboxed.cols, letterboxed.rows);

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    if (rgb.empty())
    {
        printf("Color conversion failed!\n");
        return -1;
    }

    printf("inference_detect_model: setting input\n");

    int img_width = img.cols;
    int img_height = img.rows;
    float gain = static_cast<float>(app_ctx->model_width) / std::max(img_width, img_height);
    float pad_w = (app_ctx->model_width - img_width * gain) / 2;
    float pad_h = (app_ctx->model_height - img_height * gain) / 2;

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf = rgb.data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    outputs[0].want_float = 0;
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    int8_t *output_int8 = (int8_t *)outputs[0].buf;
    int output_size = outputs[0].size;
    
    float output_scale = app_ctx->output_attrs[0].scale;
    int output_zp = app_ctx->output_attrs[0].zp;
    
    printf("output scale: %.6f, zp: %d\n", output_scale, output_zp);
    
    int num_anchors = output_size / 16;
    if (num_anchors <= 0 || num_anchors > 100000)
    {
        printf("Invalid num_anchors: %d, output_size: %d\n", num_anchors, output_size);
        return -1;
    }

    float *output_data = new float[output_size];
    for (int i = 0; i < output_size; i++)
    {
        output_data[i] = (output_int8[i] - output_zp) * output_scale;
    }

    printf("num_anchors: %d, output_size: %d\n", num_anchors, output_size);
    
    float max_obj_conf = 0.0f;
    int max_obj_idx = 0;
    for (int i = 0; i < num_anchors; i++)
    {
        if (output_data[i * 16 + 4] > max_obj_conf)
        {
            max_obj_conf = output_data[i * 16 + 4];
            max_obj_idx = i;
        }
    }
    printf("Max obj_conf: %.4f at anchor %d\n", max_obj_conf, max_obj_idx);
    printf("First 16 values: ");
    for (int i = 0; i < 16 && i < output_size; i++)
    {
        printf("%.4f ", output_data[i]);
    }
    printf("\n");

    std::vector<detect_box_t> boxes;
    std::vector<std::vector<landmark_point_t>> all_landmarks;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    int valid_count = 0;
    for (int i = 0; i < num_anchors; i++)
    {
        const float *ptr = output_data + i * 16;
        
        if (i * 16 + 14 >= output_size)
        {
            printf("Warning: index %d out of bounds\n", i * 16 + 14);
            break;
        }
        
        float cx = ptr[0];
        float cy = ptr[1];
        float w = ptr[2];
        float h = ptr[3];
        float obj_conf = sigmoid(ptr[4]);

        int class_id = 0;
        float class_conf = 0.0f;
        
        for (int c = 0; c < 3; c++)
        {
            float cls_conf = sigmoid(ptr[13 + c]);
            if (cls_conf > class_conf)
            {
                class_conf = cls_conf;
                class_id = c;
            }
        }
        float conf = obj_conf * class_conf;

        if (valid_count < 10 && obj_conf > 0.01f)
        {
            printf("Anchor %d: raw_obj=%.4f, obj_conf=%.4f, cls[0]=%.4f, cls[1]=%.4f, cls[2]=%.4f, class_conf=%.4f, conf=%.4f\n",
                   i, ptr[4], obj_conf, sigmoid(ptr[13]), sigmoid(ptr[14]), sigmoid(ptr[15]), class_conf, conf);
            valid_count++;
        }

        if (conf < conf_threshold)
            continue;

        float x1 = (cx - w / 2 - pad_w) / gain;
        float y1 = (cy - h / 2 - pad_h) / gain;
        float x2 = (cx + w / 2 - pad_w) / gain;
        float y2 = (cy + h / 2 - pad_h) / gain;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(img_width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(img_height)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(img_width)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(img_height)));

        detect_box_t box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;
        box.conf = conf;
        box.cls_id = class_id;

        std::vector<landmark_point_t> landmarks(4);
        for (int j = 0; j < 4; j++)
        {
            float lx = (ptr[5 + j * 2] - pad_w) / gain;
            float ly = (ptr[5 + j * 2 + 1] - pad_h) / gain;
            landmarks[j].x = std::max(0.0f, std::min(lx, static_cast<float>(img_width)));
            landmarks[j].y = std::max(0.0f, std::min(ly, static_cast<float>(img_height)));
        }

        boxes.push_back(box);
        all_landmarks.push_back(landmarks);
        class_ids.push_back(class_id);
        confidences.push_back(conf);
    }

    std::vector<int> keep = nms(boxes, nms_threshold);

    det_results->count = 0;
    for (int idx : keep)
    {
        if (det_results->count >= OBJ_NUMB_MAX_SIZE)
            break;

        detect_result_t &result = det_results->results[det_results->count];
        result.box = boxes[idx];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        for (int j = 0; j < 4; j++)
        {
            result.landmarks[j] = all_landmarks[idx][j];
        }
        det_results->count++;
    }

    delete[] output_data;
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

    return 0;
}

cv::Mat letterbox(cv::Mat &img, int new_shape, cv::Scalar color)
{
    int h = img.rows;
    int w = img.cols;
    float r = static_cast<float>(new_shape) / std::max(h, w);

    int new_h = static_cast<int>(h * r);
    int new_w = static_cast<int>(w * r);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0,
               r < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);

    int dw = new_shape - new_w;
    int dh = new_shape - new_h;
    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return padded;
}

cv::Mat perspective_transform(cv::Mat &img, const std::vector<cv::Point2f> &points)
{
    if (points.size() != 4)
        return img.clone();

    std::vector<cv::Point2f> rect = order_points(points);
    const cv::Point2f &tl = rect[0];
    const cv::Point2f &tr = rect[1];
    const cv::Point2f &br = rect[2];
    const cv::Point2f &bl = rect[3];

    float width_a = std::hypot(br.x - bl.x, br.y - bl.y);
    float width_b = std::hypot(tr.x - tl.x, tr.y - tl.y);
    float height_a = std::hypot(tr.x - br.x, tr.y - br.y);
    float height_b = std::hypot(tl.x - bl.x, tl.y - bl.y);

    int max_width = std::max(1, static_cast<int>(std::round(std::max(width_a, width_b))));
    int max_height = std::max(1, static_cast<int>(std::round(std::max(height_a, height_b))));

    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(max_width - 1), 0),
        cv::Point2f(static_cast<float>(max_width - 1), static_cast<float>(max_height - 1)),
        cv::Point2f(0, static_cast<float>(max_height - 1))
    };

    cv::Mat M = cv::getPerspectiveTransform(rect, dst_points);
    cv::Mat warped;
    cv::warpPerspective(img, warped, M, cv::Size(max_width, max_height));
    return warped;
}

cv::Mat split_double_plate(cv::Mat &plate_img)
{
    int h = plate_img.rows;
    int w = plate_img.cols;

    cv::Mat upper = plate_img(cv::Range(0, static_cast<int>(5.0 / 12 * h)), cv::Range::all());
    cv::Mat lower = plate_img(cv::Range(static_cast<int>(1.0 / 3 * h), h), cv::Range::all());

    cv::Mat upper_resized;
    cv::resize(upper, upper_resized, cv::Size(lower.cols, lower.rows));

    cv::Mat result;
    cv::hconcat(upper_resized, lower, result);
    return result;
}

void draw_results(cv::Mat &img, const std::vector<final_result_t> &results)
{
    printf("draw_results: enter, results count = %zu, img size = %dx%d\n", results.size(), img.cols, img.rows);
    fflush(stdout);
    
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 255, 0)
    };

    if (results.empty())
    {
        printf("draw_results: no results to draw\n");
        fflush(stdout);
        return;
    }

    for (size_t idx = 0; idx < results.size(); idx++)
    {
        const auto &res = results[idx];
        printf("draw_results: processing result %zu, class_type = %d\n", idx, res.class_type);
        fflush(stdout);
        
        int color_idx = std::min(res.class_type, static_cast<int>(colors.size() - 1));
        printf("draw_results: drawing rectangle\n");
        fflush(stdout);
        cv::rectangle(img, res.rect, colors[color_idx], 2);
        printf("draw_results: rectangle done\n");
        fflush(stdout);

        std::string label;
        if (res.class_type == 2)
        {
            label = std::string("车辆颜色:") + to_car_color_cn_draw(res.car_color);
        }
        else
        {
            label = res.plate_number + " " + to_plate_color_cn_draw(res.plate_color);
            if (res.class_type == 1)
            {
                label += " 双层";
            }

            for (const auto &pt : res.landmarks)
            {
                cv::circle(img, pt, 3, cv::Scalar(255, 0, 0), -1);
            }
        }

        printf("draw_results: putting text: %s\n", label.c_str());
        int text_height = std::max(18, res.class_type == 2 ? res.rect.height / 18 : res.rect.height / 2);
        int text_x = std::max(0, res.rect.x);
        int text_y = std::max(0, res.rect.y - text_height - 4);
        draw_utf8_text(img, label, cv::Point(text_x, text_y), cv::Scalar(0, 255, 0), text_height);
        printf("draw_results: text done\n");
    }
    printf("draw_results: all done\n");
}
