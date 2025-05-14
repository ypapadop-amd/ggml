#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef GGML_USE_HSA
#include "ggml-hsa.h"
#endif
#ifdef GGML_USE_HIP
#include "ggml-cuda.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

constexpr int N_THREADS = 4;
constexpr size_t DEFAULT_BUFFER_SIZE = 8192;

class Matrix {
public:
  Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols) {}

  Matrix(int rows, int cols, std::initializer_list<float> init)
      : rows_(rows), cols_(cols), data_(init.begin(), init.end()) {
    assert(init.size() == size_t(rows_) * size_t(cols_));
  }

  Matrix(int rows, int cols, const std::vector<float> &v)
      : rows_(rows), cols_(cols), data_(v) {
    assert(v.size() == size_t(rows_) * size_t(cols_));
  }

  int rows() const { return rows_; }
  int cols() const { return cols_; }

  float *dataPtr() { return data_.data(); }
  const float *dataPtr() const { return data_.data(); }

  bool equals(const Matrix &other, float tol = Matrix::tol) const {
    if (rows_ != other.rows_ || cols_ != other.cols_)
      return false;
    for (size_t i = 0; i < data_.size(); ++i) {
      if (std::fabs(data_[i] - other.data_[i]) > tol) {
        return false;
      }
    }
    return true;
  }

  void print() const {
    for (int r = 0; r < rows_; ++r) {
      for (int c = 0; c < cols_; ++c) {
        printf("%.2f ", data_[r * cols_ + c]);
      }
      printf("\n");
    }
  }

  void verify(const Matrix &expected, float tol = Matrix::tol) const {
    if (!equals(expected, tol)) {
      printf("Verification FAILED!\n");
      printf("Computed (%dx%d):\n", rows_, cols_);
      print();
      printf("Expected (%dx%d):\n", expected.rows(), expected.cols());
      expected.print();
    } else {
      printf("Verification PASSED for %dx%d matrix.\n", rows_, cols_);
    }
  }

private:
  int rows_, cols_;
  std::vector<float> data_;
  static constexpr float tol = 1e-5f;
};

class BackendManager {
public:
  BackendManager() {
    initCpu();
    initHsa();
    initHip();

    // add the cpu backend to the end of the list as a fallback device
    addUnique(cpu_);
  }

  ggml_backend_t getCpu() const { return cpu_; }
  ggml_backend_t getHsa() const { return hsa_; }
  ggml_backend_t getHip() const { return hip_; }

  const std::vector<ggml_backend_t> &all() const { return backends_; }
  ggml_backend_sched_t createScheduler(size_t graphSize) const {
    return ggml_backend_sched_new(
        const_cast<ggml_backend_t *>(backends_.data()), nullptr,
        backends_.size(), graphSize, false);
  }

private:
  void initCpu() {
    cpu_ = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu_, N_THREADS);
  }

  void initHsa() {
#ifdef GGML_USE_HSA
    hsa_ = ggml_backend_hsa_init(0);
    if (!hsa_) {
      fprintf(stderr, "HSA init failed, fallback to CPU\n");
      hsa_ = cpu_;
    } else {
      addUnique(hsa_);
    }
#else
    hsa_ = cpu_;
    printf("HSA not configured, using CPU\n");
#endif
  }

  void initHip() {
#ifdef GGML_USE_HIP
    hip_ = ggml_backend_cuda_init(0);
    if (!hip_) {
      fprintf(stderr, "HIP init failed, fallback to CPU\n");
      hip_ = cpu_;
    } else {
      addUnique(hip_);
    }
#else
    hip_ = cpu_;
    printf("HIP not configured, using CPU\n");
#endif
  }

  void addUnique(ggml_backend_t b) {
    for (auto &x : backends_)
      if (x == b)
        return;
    backends_.push_back(b);
  }

  ggml_backend_t cpu_{nullptr}, hsa_{nullptr}, hip_{nullptr};
  std::vector<ggml_backend_t> backends_;
};

class BackendBinding {
public:
  BackendBinding(ggml_backend_t bk, size_t buffer_size)
      : backend_(bk), buffer_(ggml_backend_alloc_buffer(bk, buffer_size)),
        alloc_(ggml_tallocr_new(buffer_)) {}

  ~BackendBinding() {
    if (buffer_) {
      ggml_backend_buffer_free(buffer_);
    }
  }

  ggml_backend_t getBackend() const { return backend_; }

  void bindTensor(ggml_tensor *t) const { ggml_tallocr_alloc(&alloc_, t); }

private:
  ggml_backend_t backend_;
  ggml_backend_buffer_t buffer_;
  mutable ggml_tallocr alloc_;
};

class SimpleModel {
public:
  SimpleModel(const BackendManager &mgr, const Matrix &A0, const Matrix &B0,
              const Matrix &A1, const Matrix &B1)
      : mgr_(mgr), rowsA(A0.rows()), colsA(A0.cols()), rowsB(B0.rows()),
        colsB(B0.cols()),
        ctx_(ggml_init({ggml_tensor_overhead() * tensorCount(), nullptr, true}),
             &ggml_free),
        hsaBinding_(mgr.getHsa(), DEFAULT_BUFFER_SIZE),
        hipBinding_(mgr.getHip(), DEFAULT_BUFFER_SIZE) {
    // create tensors
    a0_ = newTensor(colsA, rowsA);
    b0_ = newTensor(colsB, rowsB);
    a1_ = newTensor(colsA, rowsA);
    b1_ = newTensor(colsB, rowsB);

    // bind tensors to backend
    hsaBinding_.bindTensor(a0_);
    hsaBinding_.bindTensor(b0_);
    hipBinding_.bindTensor(a1_);
    hipBinding_.bindTensor(b1_);

    // upload the data immediately
    setData(a0_, A0.dataPtr());
    setData(b0_, B0.dataPtr());
    setData(a1_, A1.dataPtr());
    setData(b1_, B1.dataPtr());
  }

  ggml_tensor *a0() const { return a0_; }
  ggml_tensor *b0() const { return b0_; }
  ggml_tensor *a1() const { return a1_; }
  ggml_tensor *b1() const { return b1_; }
  ggml_context *ctx() const { return ctx_.get(); }

private:
  static constexpr int tensorCount() { return 4; }

  ggml_tensor *newTensor(int c, int r) {
    return ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, c, r);
  }

  void setData(ggml_tensor *t, const float *d) {
    ggml_backend_tensor_set(t, const_cast<float *>(d), 0, ggml_nbytes(t));
  }

  const BackendManager &mgr_;
  BackendBinding hsaBinding_;
  BackendBinding hipBinding_;
  int rowsA, colsA, rowsB, colsB;
  std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx_{nullptr, &ggml_free};
  ggml_tensor *a0_, *b0_, *a1_, *b1_;
};

class GraphBuilder {
public:
  GraphBuilder() {
    size_t tensorOverhead = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE;
    size_t graphOverhead = ggml_graph_overhead();
    buffer_.resize(tensorOverhead + graphOverhead);
  }

  ggml_cgraph *build(const SimpleModel &m) const {
    ggml_init_params p{buffer_.size(), buffer_.data(), true};
    auto ctx0 = ggml_init(p);
    auto gf = ggml_new_graph(ctx0);
    auto r0 = ggml_mul_mat(ctx0, m.a0(), m.b0());
    auto r1 = ggml_mul_mat(ctx0, m.a1(), m.b1());
    auto r2 = ggml_mul_mat(ctx0, r0, r1);
    ggml_build_forward_expand(gf, r2);
    ggml_free(ctx0);
    return gf;
  }

private:
  mutable std::vector<uint8_t> buffer_;
};

int main() {
  ggml_time_init();

  const int rows_A = 5, cols_A = 3;
  const int rows_B = 4, cols_B = 3;

  Matrix A0(rows_A, cols_A,
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  Matrix A1(rows_A, cols_A, {5, 4, 3, 6, 7, 8, 9, 1, 2, 3, 5, 7, 8, 6, 4});

  Matrix B0(rows_B, cols_B, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24});

  Matrix B1(rows_B, cols_B, {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23});

  BackendManager mgr;
  SimpleModel model(mgr, A0, B0, A1, B1);

  auto sched = mgr.createScheduler(GGML_DEFAULT_GRAPH_SIZE);
  GraphBuilder builder;
  auto graph = builder.build(model);

  ggml_backend_sched_reset(sched);
  ggml_backend_sched_graph_compute(sched, graph);

  auto *c0 = ggml_graph_node(graph, -3);
  auto *c1 = ggml_graph_node(graph, -2);
  auto *out = ggml_graph_node(graph, -1);

  std::vector<float> out0(ggml_nelements(c0));
  std::vector<float> out1(ggml_nelements(c1));
  std::vector<float> out2(ggml_nelements(out));

  ggml_backend_tensor_get(c0, out0.data(), 0, ggml_nbytes(c0));
  ggml_backend_tensor_get(c1, out1.data(), 0, ggml_nbytes(c1));
  ggml_backend_tensor_get(out, out2.data(), 0, ggml_nbytes(out));

  Matrix C0(c0->ne[1], c0->ne[0], out0);
  Matrix C0_expected(4, 5, {28,  64,  100, 136, 172, 64,  154, 244, 334, 424,
                            100, 244, 388, 532, 676, 136, 334, 532, 730, 928});
  C0.verify(C0_expected);

  Matrix C1(c1->ne[1], c1->ne[0], out1);
  Matrix C1_expected(4, 5,
                     {32,  67,  22,  53,  46,  104, 193, 94,  143, 154, 176,
                      319, 166, 233, 262, 248, 445, 238, 323, 370

                     });
  C1.verify(C1_expected);

  Matrix final_result(out->ne[1], out->ne[0], out2);
  Matrix final_result_expected(4, 4,
                               {22504, 54940, 87376, 119812, 70600, 172372,
                                274144, 375916, 118696, 289804, 460912, 632020,
                                166792, 407236, 647680, 888124});
  final_result.verify(final_result_expected);

  return 0;
}
