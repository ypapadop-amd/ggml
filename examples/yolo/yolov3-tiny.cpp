#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "yolo-image.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <algorithm>
#include <thread>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct conv2d_layer {
    struct ggml_tensor * weights;
    struct ggml_tensor * biases;
    struct ggml_tensor * scales;
    struct ggml_tensor * rolling_mean;
    struct ggml_tensor * rolling_variance;
    int padding = 1;
    bool batch_normalize = true;
    bool activate = true; // true for leaky relu, false for linear
};

struct yolo_model {
    int width = 416;
    int height = 416;
    std::vector<conv2d_layer> conv2d_layers;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

struct yolo_layer {
    int classes = 80;
    std::vector<int> mask;
    std::vector<float> anchors;
    std::vector<float> predictions;
    int w;
    int h;

    yolo_layer(int classes, const std::vector<int> & mask, const std::vector<float> & anchors, struct ggml_tensor * prev_layer)
        : classes(classes), mask(mask), anchors(anchors)
    {
        w = prev_layer->ne[0];
        h = prev_layer->ne[1];
        predictions.resize(ggml_nbytes(prev_layer)/sizeof(float));
        ggml_backend_tensor_get(prev_layer, predictions.data(), 0, ggml_nbytes(prev_layer));
    }

    int entry_index(int location, int entry) const {
        int n = location / (w*h);
        int loc = location % (w*h);
        return n*w*h*(4+classes+1) + entry*w*h + loc;
    }
};

struct box {
    float x, y, w, h;
};

struct detection {
    box bbox;
    std::vector<float> prob;
    float objectness;
};

static bool load_model(const std::string & fname, yolo_model & model) {
    struct ggml_context * tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    int num_tensors = gguf_get_n_tensors(gguf_ctx);
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor * dst = ggml_dup_tensor(model.ctx, src);
        ggml_set_name(dst, name);
    }
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // copy tensors from main memory to backend
    for (struct ggml_tensor * cur = ggml_get_first_tensor(model.ctx); cur != NULL; cur = ggml_get_next_tensor(model.ctx, cur)) {
        struct ggml_tensor * src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    gguf_free(gguf_ctx);

    model.width  = 416;
    model.height = 416;
    model.conv2d_layers.resize(13);
    model.conv2d_layers[7].padding = 0;
    model.conv2d_layers[9].padding = 0;
    model.conv2d_layers[9].batch_normalize = false;
    model.conv2d_layers[9].activate = false;
    model.conv2d_layers[10].padding = 0;
    model.conv2d_layers[12].padding = 0;
    model.conv2d_layers[12].batch_normalize = false;
    model.conv2d_layers[12].activate = false;
    for (int i = 0; i < (int)model.conv2d_layers.size(); i++) {
        char name[256];
        snprintf(name, sizeof(name), "l%d_weights", i);
        model.conv2d_layers[i].weights = ggml_get_tensor(model.ctx, name);
        snprintf(name, sizeof(name), "l%d_biases", i);
        model.conv2d_layers[i].biases = ggml_get_tensor(model.ctx, name);
        if (model.conv2d_layers[i].batch_normalize) {
            snprintf(name, sizeof(name), "l%d_scales", i);
            model.conv2d_layers[i].scales = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "l%d_rolling_mean", i);
            model.conv2d_layers[i].rolling_mean = ggml_get_tensor(model.ctx, name);
            snprintf(name, sizeof(name), "l%d_rolling_variance", i);
            model.conv2d_layers[i].rolling_variance = ggml_get_tensor(model.ctx, name);
        }
    }
    return true;
}

static bool load_labels(const char * filename, std::vector<std::string> & labels)
{
    std::ifstream file_in(filename);
    if (!file_in) {
        return false;
    }
    std::string line;
    while (std::getline(file_in, line)) {
        labels.push_back(line);
    }
    GGML_ASSERT(labels.size() == 80);
    return true;
}

static bool load_alphabet(std::vector<yolo_image> & alphabet)
{
    alphabet.resize(8 * 128);
    for (int j = 0; j < 8; j++) {
        for (int i = 32; i < 127; i++) {
            char fname[256];
            snprintf(fname, sizeof(fname), "data/labels/%d_%d.png", i, j);
            if (!load_image(fname, alphabet[j*128 + i])) {
                fprintf(stderr, "Cannot load '%s'\n", fname);
                return false;
            }
        }
    }
    return true;
}

static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer)
{
    struct ggml_tensor * result = ggml_conv_2d(ctx, layer.weights, input, 1, 1, layer.padding, layer.padding, 1, 1);
    if (layer.batch_normalize) {
        result = ggml_sub(ctx, result, ggml_repeat(ctx, layer.rolling_mean, result));
        result = ggml_div(ctx, result, ggml_sqrt(ctx, ggml_repeat(ctx, layer.rolling_variance, result)));
        result = ggml_mul(ctx, result, ggml_repeat(ctx, layer.scales, result));
    }
    result = ggml_add(ctx, result, ggml_repeat(ctx, layer.biases, result));
    if (layer.activate) {
        result = ggml_leaky_relu(ctx, result, 0.1f, true);
    }
    return result;
}

static void activate_array(float * x, const int n)
{
    // logistic activation
    for (int i = 0; i < n; i++) {
        x[i] = 1./(1. + exp(-x[i]));
    }
}

static void apply_yolo(yolo_layer & layer)
{
    int w = layer.w;
    int h = layer.h;
    int N = layer.mask.size();
    float * data = layer.predictions.data();
    for (int n = 0; n < N; n++) {
        int index = layer.entry_index(n*w*h, 0);
        activate_array(data + index, 2*w*h);
        index = layer.entry_index(n*w*h, 4);
        activate_array(data + index, (1+layer.classes)*w*h);
    }
}

static box get_yolo_box(const yolo_layer & layer, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    const float * predictions = layer.predictions.data();
    box b;
    b.x = (i + predictions[index + 0*stride]) / lw;
    b.y = (j + predictions[index + 1*stride]) / lh;
    b.w = exp(predictions[index + 2*stride]) * layer.anchors[2*n]   / w;
    b.h = exp(predictions[index + 3*stride]) * layer.anchors[2*n+1] / h;
    return b;
}

static void correct_yolo_box(box & b, int im_w, int im_h, int net_w, int net_h)
{
    int new_w = 0;
    int new_h = 0;
    if (((float)net_w/im_w) < ((float)net_h/im_h)) {
        new_w = net_w;
        new_h = (im_h * net_w)/im_w;
    } else {
        new_h = net_h;
        new_w = (im_w * net_h)/im_h;
    }
    b.x = (b.x - (net_w - new_w)/2./net_w) / ((float)new_w/net_w);
    b.y = (b.y - (net_h - new_h)/2./net_h) / ((float)new_h/net_h);
    b.w *= (float)net_w/new_w;
    b.h *= (float)net_h/new_h;
}

static void get_yolo_detections(const yolo_layer & layer, std::vector<detection> & detections, int im_w, int im_h, int netw, int neth, float thresh)
{
    int w = layer.w;
    int h = layer.h;
    int N = layer.mask.size();
    const float * predictions = layer.predictions.data();
    std::vector<detection> result;
    for (int i = 0; i < w*h; i++) {
        for (int n = 0; n < N; n++) {
            int obj_index = layer.entry_index(n*w*h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) {
                continue;
            }
            detection det;
            int box_index = layer.entry_index(n*w*h + i, 0);
            int row = i / w;
            int col = i % w;
            det.bbox = get_yolo_box(layer, layer.mask[n], box_index, col, row, w, h, netw, neth, w*h);
            correct_yolo_box(det.bbox, im_w, im_h, netw, neth);
            det.objectness = objectness;
            det.prob.resize(layer.classes);
            for (int j = 0; j < layer.classes; j++) {
                int class_index = layer.entry_index(n*w*h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                det.prob[j] = (prob > thresh) ? prob : 0;
            }
            detections.push_back(det);
        }
    }
}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(const box & a, const box & b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

static float box_union(const box & a, const box & b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

static float box_iou(const box & a, const box & b)
{
    return box_intersection(a, b)/box_union(a, b);
}

static void do_nms_sort(std::vector<detection> & dets, int classes, float thresh)
{
    int k = (int)dets.size()-1;
    for (int i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            std::swap(dets[i], dets[k]);
            --k;
            --i;
        }
    }
    int total = k+1;
    for (int k = 0; k < classes; ++k) {
        std::sort(dets.begin(), dets.begin()+total, [=](const detection & a, const detection & b) {
            return a.prob[k] > b.prob[k];
        });
        for (int i = 0; i < total; ++i) {
            if (dets[i].prob[k] == 0) {
                continue;
            }
            box a = dets[i].bbox;
            for (int j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

static float get_color(int c, int x, int max)
{
    float colors[6][3] = { {1,0,1}, {0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,0} };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    return r;
}

static void draw_detections(yolo_image & im, const std::vector<detection> & dets, float thresh, const std::vector<std::string> & labels, const std::vector<yolo_image> & alphabet)
{
    int classes = (int)labels.size();
    for (int i = 0; i < (int)dets.size(); i++) {
        std::string labelstr;
        int cl = -1;
        for (int j = 0; j < (int)dets[i].prob.size(); j++) {
            if (dets[i].prob[j] > thresh) {
                if (cl < 0) {
                    labelstr = labels[j];
                    cl = j;
                } else {
                    labelstr += ", ";
                    labelstr += labels[j];
                }
                printf("%s: %.0f%%\n", labels[j].c_str(), dets[i].prob[j]*100);
            }
        }
        if (cl >= 0) {
            int width = im.h * .006;
            int offset = cl*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = dets[i].bbox;

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if (left < 0) left = 0;
            if (right > im.w-1) right = im.w-1;
            if (top < 0) top = 0;
            if (bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            yolo_image label = get_label(alphabet, labelstr, (im.h*.03));
            draw_label(im, top + width, left, label, rgb);
        }
    }
}

static void print_shape(int layer, const ggml_tensor * t)
{
    printf("Layer %2d output shape:  %3d x %3d x %4d x %3d\n", layer, (int)t->ne[0], (int)t->ne[1], (int)t->ne[2], (int)t->ne[3]);
}

static struct ggml_cgraph * build_graph(struct ggml_context * ctx_cgraph, const yolo_model & model) {
    struct ggml_cgraph * gf = ggml_new_graph(ctx_cgraph);

    struct ggml_tensor * input = ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, model.width, model.height, 3, 1);
    ggml_set_name(input, "input");
    struct ggml_tensor * result = apply_conv2d(ctx_cgraph, input, model.conv2d_layers[0]);
    print_shape(0, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(1, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[1]);
    print_shape(2, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(3, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[2]);
    print_shape(4, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(5, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[3]);
    print_shape(6, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(7, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[4]);
    struct ggml_tensor * layer_8 = result;
    print_shape(8, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    print_shape(9, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[5]);
    print_shape(10, result);
    result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0.5, 0.5);
    print_shape(11, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[6]);
    print_shape(12, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[7]);
    struct ggml_tensor * layer_13 = result;
    print_shape(13, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[8]);
    print_shape(14, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[9]);
    struct ggml_tensor * layer_15 = result;
    ggml_set_output(layer_15);
    ggml_set_name(layer_15, "layer_15");

    print_shape(15, result);
    result = apply_conv2d(ctx_cgraph, layer_13, model.conv2d_layers[10]);
    print_shape(18, result);
    result = ggml_upscale(ctx_cgraph, result, 2, GGML_SCALE_MODE_NEAREST);
    print_shape(19, result);
    result = ggml_concat(ctx_cgraph, result, layer_8, 2);
    print_shape(20, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[11]);
    print_shape(21, result);
    result = apply_conv2d(ctx_cgraph, result, model.conv2d_layers[12]);
    struct ggml_tensor * layer_22 = result;
    ggml_set_output(layer_22);
    ggml_set_name(layer_22, "layer_22");
    print_shape(22, result);

    ggml_build_forward_expand(gf, layer_15);
    ggml_build_forward_expand(gf, layer_22);
    return gf;
}

void detect(yolo_image & img, struct ggml_cgraph * gf, const yolo_model & model, float thresh, const std::vector<std::string> & labels, const std::vector<yolo_image> & alphabet)
{
    std::vector<detection> detections;
    yolo_image sized = letterbox_image(img, model.width, model.height);
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(input, sized.data.data(), 0, ggml_nbytes(input));

    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() failed\n", __func__);
        return;
    }

    struct ggml_tensor * layer_15 = ggml_graph_get_tensor(gf, "layer_15");
    yolo_layer yolo16{ 80, {3, 4, 5}, {10, 14, 23, 27, 37,58, 81, 82, 135, 169, 344, 319}, layer_15};
    apply_yolo(yolo16);
    get_yolo_detections(yolo16, detections, img.w, img.h, model.width, model.height, thresh);

    struct ggml_tensor * layer_22 = ggml_graph_get_tensor(gf, "layer_22");
    yolo_layer yolo23{ 80, {0, 1, 2}, {10, 14, 23, 27, 37,58, 81, 82, 135, 169, 344, 319}, layer_22};
    apply_yolo(yolo23);
    get_yolo_detections(yolo23, detections, img.w, img.h, model.width, model.height, thresh);

    do_nms_sort(detections, yolo23.classes, .45);
    draw_detections(img, detections, thresh, labels, alphabet);
}

struct yolo_params {
    float thresh          = 0.5;
    std::string model     = "yolov3-tiny.gguf";
    std::string fname_inp = "input.jpg";
    std::string fname_out = "predictions.jpg";
    int         n_threads  = std::max(1U, std::thread::hardware_concurrency()/2);
    std::string device;
};

void yolo_print_usage(int argc, char ** argv, const yolo_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,  --help                show this help message and exit\n");
    fprintf(stderr, "  -d,  --device DEV          device to use\n");
    fprintf(stderr, "  -t,  --threads N           number of threads for the CPU backend (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -th, --thresh T            detection threshold (default: %.2f)\n", params.thresh);
    fprintf(stderr, "  -m,  --model FNAME         model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -i,  --inp FNAME           input file (default: %s)\n", params.fname_inp.c_str());
    fprintf(stderr, "  -o,  --out FNAME           output file (default: %s)\n", params.fname_out.c_str());
    fprintf(stderr, "\n");
}

bool yolo_params_parse(int argc, char ** argv, yolo_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-th" || arg == "--thresh") {
            params.thresh = std::stof(argv[++i]);
            if (params.thresh < 0 || params.thresh > 1) {
                fprintf(stderr, "error: invalid threshold: %.2f\n", params.thresh);
                return false;
            }
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-i" || arg == "--inp") {
            params.fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--out") {
            params.fname_out = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) {
                return false;
            }
            params.device = argv[i];
            if (ggml_backend_dev_by_name(params.device.c_str()) == nullptr) {
                fprintf(stderr, "error: unknown device: %s\n", params.device.c_str());
                fprintf(stderr, "available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                }
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            yolo_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            yolo_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

static ggml_backend_t create_backend(const yolo_params & params) {
    ggml_backend_t backend = nullptr;

    if (!params.device.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(params.device.c_str());
        if (dev) {
            backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                fprintf(stderr, "Failed to create backend for device %s\n", params.device.c_str());
                return nullptr;
            }
        }
    }

    // try to initialize a GPU backend first
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    // if there aren't GPU backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    if (backend) {
        fprintf(stderr, "%s: using %s backend\n", __func__, ggml_backend_name(backend));

        // set the number of threads
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, params.n_threads);
            }
        }
    }

    return backend;
}

int main(int argc, char *argv[])
{
    ggml_backend_load_all();
    ggml_time_init();
    yolo_model model;

    yolo_params params;
    if (!yolo_params_parse(argc, argv, params)) {
        return 1;
    }
    model.backend = create_backend(params);
    if (!model.backend) {
        fprintf(stderr, "Failed to create backend\n");
        return 1;
    }

    if (!load_model(params.model, model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }
    yolo_image img(0,0,0);
    if (!load_image(params.fname_inp.c_str(), img)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    std::vector<std::string> labels;
    if (!load_labels("data/coco.names", labels)) {
        fprintf(stderr, "%s: failed to load labels from 'data/coco.names'\n", __func__);
        return 1;
    }
    std::vector<yolo_image> alphabet;
    if (!load_alphabet(alphabet)) {
        fprintf(stderr, "%s: failed to load alphabet\n", __func__);
        return 1;
    }

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context * ctx_cgraph = ggml_init(params0);
    struct ggml_cgraph * gf = build_graph(ctx_cgraph, model);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    const int64_t t_start_ms = ggml_time_ms();
    detect(img, gf, model, params.thresh, labels, alphabet);
    const int64_t t_detect_ms = ggml_time_ms() - t_start_ms;
    if (!save_image(img, params.fname_out.c_str(), 80)) {
        fprintf(stderr, "%s: failed to save image to '%s'\n", __func__, params.fname_out.c_str());
        return 1;
    }
    printf("Detected objects saved in '%s' (time: %f sec.)\n", params.fname_out.c_str(), t_detect_ms / 1000.0f);

    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
