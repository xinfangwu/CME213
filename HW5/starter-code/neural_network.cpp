#define ARMA_ALLOW_FAKE_GCC
#include "neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mpi.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                                 Utilities                                  //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

void label_to_y(arma::Row<nn_real> label, int num_classes,
                arma::Mat<nn_real> &y)
{
  y.set_size(num_classes, label.size());
  y.fill(0);
  for (int i = 0; i < label.size(); ++i)
  {
    assert(label(i) >= 0);
    assert(label(i) < num_classes);
    y(label(i), i) = 1;
  }
}

int get_num_batches(int N, int batch_size)
{
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch)
{
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             CPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

NeuralNetwork::NeuralNetwork(std::vector<int> &_H)
{
  W.resize(num_layers);
  b.resize(num_layers);
  H = _H;

  assert(H.size() == 3);

  for (int i = 0; i < num_layers; i++)
  {
    arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
    assert(i + 1 < 3);
    W[i] = 0.01 * arma::randn<arma::Mat<nn_real>>(H[i + 1], H[i]);
    b[i].zeros(H[i + 1]);
  }
}

void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1. / (1. + arma::exp(-mat));
}

void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

void forward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;
  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }
  return norm_sum;
}

nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  return loss;
}

void backward(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, hyperparameters &hparams)
{
  int N = X.n_cols;
  int iter = 0;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, hparams.batch_size);

  for (int epoch = 0; epoch < hparams.num_epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = batch_start + get_batch_size(N, hparams.batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      assert(batch < num_batches - 1 || last_col == X.n_cols);

      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);

      struct cache bpcache;
      forward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backward(nn, y_batch, hparams.reg, bpcache, bpgrads);

      if (iter % 1 == 0)
      {
        printf("Seq loss at iteration %d of epoch %d/%d = %25.20f\n", iter,
               epoch, hparams.num_epochs,
               loss(nn, bpcache.yc, y_batch, hparams.reg));
      }

      // Optimizer step
      for (int i = 0; i < nn.W.size(); ++i)
        nn.W[i] -= hparams.learning_rate * bpgrads.dW[i];

      for (int i = 0; i < nn.b.size(); ++i)
        nn.b[i] -= hparams.learning_rate * bpgrads.db[i];

      batch_start = last_col;
      iter++;
    }
  }
}

// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //
//                             GPU implementation                             //
// +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+- //

GPUGrads::GPUGrads(const NeuralNetwork &nn)
{
  total_size = 0;
  /**
   * TODO: We need to allocate memory for DeviceMatrix objects that will be
   * stored in the vectors dW and db. For this, we use gpu_mem_pool. The space
   * allocated with gpu_mem_pool will be used to set up dW and db. Calculate the
   * total number of elements that we need to allocate memory for.
   */
  for (size_t i = 0; i < nn.num_layers - 1; i++) {
    total_size += nn.H[i] * nn.H[i + 1]; 
    total_size += nn.H[i + 1];           
  }

  gpu_mem_pool = std::make_shared<DeviceAllocator>(total_size * sizeof(nn_real));
  nn_real *gpu_mem_pool_start = gpu_mem_pool.get()->memptr();

  size_t offset = 0;
  dW.resize(nn.num_layers);
  db.resize(nn.num_layers);
  /**
   * TODO: We have allocated memory for all the DeviceMatrix objects that will
   * be stored in dW and db. Use the constructor
   * DeviceMatrix::DeviceMatrix(nn_real *gpu_mat, int n_rows, int n_cols) to
   * create instances of DeviceMatrix that will be stored in dW and db, then
   * store them in dW and db. E.g. dW[0] = DeviceMatrix(...). You should use the
   * space we previously allocated, gpu_mem_pool_start. Use an offset and
   * pointer arithmetic to pass the correct pointer "gpu_mat" for each
   * DeviceMatrix object that you instantiate.
   * The template pseudo-code looks like:
   * dW[...] = DeviceMatrix(gpu_mem_pool_start + ..., n_rows, n_cols);
   */

  for (size_t i = 0; i < nn.num_layers; ++i) {

    dW[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.H[i], nn.H[i + 1]);
    offset += nn.H[i] * nn.H[i + 1];

    
    db[i] = DeviceMatrix(gpu_mem_pool_start + offset, nn.H[i + 1], 1);
    offset += nn.H[i + 1];
  }
}

DataParallelNeuralNetwork::DataParallelNeuralNetwork(NeuralNetwork &nn,
                                                     nn_real reg, nn_real lr,
                                                     int batch_size,
                                                     int num_procs)
{
  /**
   * TODO: Implement this constructor.
   * NOTE: In this homework, num_procs = 1. In Homework 6, num_procs > 1, and in
   * this case the regularization parameter reg must be adjusted. For this
   * homework, your implementation of this constructor can ignore num_procs. You
   * will need to correctly initialize at least: cache.a, cache.z, W[], b[], and
   * reg.
   */

  // Initialize regularization parameter and learning rate
  this->reg = reg;
  this->lr = lr;
  this->num_procs = num_procs;

  int numLayer = nn.num_layers;
  assert(numLayer == nn.W.size());
  assert(numLayer == nn.b.size());

  W.resize(numLayer);
  b.resize(numLayer);
  // Initialize weights and biases from the given neural network model
  for(size_t i=0; i<numLayer; i++){
    // weights
    DeviceMatrix cpuW(nn.W[i]);
    W[i] = DeviceMatrix(nn.W[i].n_rows, nn.W[i].n_cols);
    W[i] = cpuW;

    // biases
    DeviceMatrix cpub(nn.b[i]);
    b[i] = DeviceMatrix(nn.b[i].n_rows, nn.b[i].n_cols);
    b[i] = cpub;
  }

  // Initialize cache 
  cache.a.resize(numLayer); 
  cache.z.resize(numLayer);

  // Initializw grad
  grads = GPUGrads(nn);
  grads.dW.resize(numLayer);
  grads.db.resize(numLayer);
}

void DataParallelNeuralNetwork::forward(const DeviceMatrix &X)
{
  /**
   * TODO: Implement this function.
   * HINT: See the CPU implementation "void forward()" above.
   * Use the CUDA kernels in "gpu_func.cu". You should not have to implement
   * any new CUDA kernel.
   * Examples of kernels to use: DRepeatColVec, tiledGEMM, DSigmoid, DSoftmax.
   */


  // assert(X.n_rows == nn.W[0].n_cols);
  // cache.X = X;
  // int N = X.n_cols;
  assert(X.n_rows == this->W[0].n_cols);
  cache.X = X;

  // Layer 1
  // arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N); 
  // cache.z[0] = z1;
  assert(this->b[0].n_cols == 1);
  DeviceMatrix z1(this->W[0].n_rows, X.n_cols);
  DRepeatColVec(this->b[0], z1, X.n_cols);
  tiledGEMM(this->W[0], false, X, false, z1, 1, 1);
  cache.z[0] = z1;

  // arma::Mat<nn_real> a1;
  // sigmoid(z1, a1);
  // cache.a[0] = a1;
  DeviceMatrix a1(z1.n_rows, z1.n_cols);
  assert(a1.n_cols == z1.n_cols && a1.n_rows == z1.n_rows);
  DSigmoid(z1, a1);
  cache.a[0] = a1;

  // assert(a1.n_rows == nn.W[1].n_cols);
  // arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  // cache.z[1] = z2;
  assert(cache.a[0].n_rows == this->W[1].n_cols);
  assert(this->b[1].n_cols == 1);
  DeviceMatrix z2(this->W[1].n_rows, a1.n_cols);
  DRepeatColVec(this->b[1], z2, a1.n_cols);
  tiledGEMM(this->W[1], false, a1, false, z2, 1, 1);
  cache.z[1] = z2;

  // arma::Mat<nn_real> a2;
  // softmax(z2, a2);
  // cache.a[1] = cache.yc = a2;
  DeviceMatrix a2(z2.n_rows, z2.n_cols);
  DSoftmax(z2, a2, 0);
  cache.a[1] = a2;
  cache.yc = a2;

}

nn_real DataParallelNeuralNetwork::loss(const DeviceMatrix &y, nn_real weight)
{
  /**
   * TODO: Implement this function.
   * HINT: See the CPU implementation "nn_real loss()" above. weight is used to
   * normalize the sum of the loss across all examples in a mini-batch by the
   * number of examples in the mini-batch. Here, weight is equivalent to 1/N in
   * the CPU implementation. Examples of CUDA kernels in "gpu_func.cu" to use:
   * DCELoss, to_cpu, DSquare, DSum.
   */

  // data loss
  // nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));
  // nn_real data_loss = ce_sum / N;
  DeviceMatrix lossMatrix(1, 1);
  DCELoss(cache.yc, y, lossMatrix);
  arma::Mat<nn_real> cpu_ce_loss(1, 1);
  lossMatrix.to_cpu(cpu_ce_loss);
  nn_real data_loss = weight * cpu_ce_loss(0, 0);
  

  // reg loss
  // nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real reg_loss = 0;
  for(size_t i =0; i<this->W.size(); i++){
    DeviceMatrix W_square(this->W[i].n_rows, this->W[i].n_cols);
    DSquare(this->W[i], W_square);

    DeviceMatrix regSumRow(this->W[i].n_rows, 1);
    DSum(W_square, regSumRow, 1, 1);
    DeviceMatrix regSum(1, 1);
    DSum(regSumRow, regSum, 1, 0);

    arma::Mat<nn_real> layer_reg(1, 1);
    regSum.to_cpu(layer_reg);
    reg_loss += layer_reg(0, 0);
  }
  
  
  reg_loss = 0.5 * this->reg * reg_loss;
  
  // nn_real loss = data_loss + reg_loss;
  nn_real loss = data_loss + reg_loss;

  return loss;
}

void DataParallelNeuralNetwork::backward(const DeviceMatrix &y,
                                         nn_real grad_weight)
{
  /**
   * TODO: Implement this function.
   * HINT: See the CPU implementation "void backward()" above. If M is an
   * instance of arma::Mat<nn_real>, then M.t() is the transpose of M. The
   * binary operator % denotes element-wise multiplication of two arma matrices.
   * You can use the function DeviceMatrix::to_gpu() to copy values from a
   * DeviceMatrix to another; e.g. cache.z[0] can be overwritten.
   * Examples of CUDA kernels to use: DElemArith, to_gpu, tiledGEMM, DSum,
   * DSigmoidBackprop.
   */

  // arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  DeviceMatrix diff(this->cache.yc.n_rows, this->cache.yc.n_cols);
  assert(this->cache.yc.n_rows == diff.n_rows && this->cache.yc.n_cols == diff.n_cols);
  this->cache.yc.to_gpu(diff);
  DElemArith(diff, y, grad_weight, -1);

  // bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  DeviceMatrix dW1(this->W[1].n_rows, this->W[1].n_cols);
  this->W[1].to_gpu(dW1);
  tiledGEMM(diff, false, this->cache.a[0], true, dW1, 1, reg);
  this->grads.dW[1] = dW1;

  // bpgrads.db[1] = arma::sum(diff, 1);
  DeviceMatrix db1(this->b[1].n_rows, this->b[1].n_cols);
  DSum(diff, db1, 1, 1);
  this->grads.db[1] = db1;

  // arma::Mat<nn_real> da1 = nn.W[1].t() * diff;
  DeviceMatrix da1(cache.a[0].n_rows, cache.a[0].n_cols);
  tiledGEMM(this->W[1], true, diff, false, da1, 1, 0);

  // arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);
  DeviceMatrix dz1(da1.n_rows, da1.n_cols);
  DSigmoidBackprop(da1, this->cache.a[0], dz1);

  // bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  DeviceMatrix dW0(this->W[0].n_rows, this->W[0].n_cols);
  this->W[0].to_gpu(dW0);
  tiledGEMM(dz1, false, this->cache.X, true, dW0, 1, reg);
  this->grads.dW[0] = dW0;

  // bpgrads.db[0] = arma::sum(dz1, 1);
  DeviceMatrix db0(this->b[0].n_rows, this->b[1].n_cols);
  DSum(dz1, db0, 1, 1);
  assert(this->grads.db[0].n_rows == db0.n_rows && this->grads.db[0].n_cols == db0.n_cols);
  this->grads.db[0] = db0;

}

void DataParallelNeuralNetwork::step()
{
  /**
   * TODO: Implement this function.
   * HINT: See part of the CPU implementation void train above.
   * Example of CUDA kernel to use: DElemArith.
   */
  
  // Optimizer step
  // for (int i = 0; i < nn.W.size(); ++i)
  //   nn.W[i] -= hparams.learning_rate * bpgrads.dW[i];
  // for (int i = 0; i < nn.b.size(); ++i)
  //   nn.b[i] -= hparams.learning_rate * bpgrads.db[i];
  assert(this->W.size() == this->b.size());
  for (int i = 0; i < this->W.size(); ++i) {
    DElemArith(this->W[i], this->grads.dW[i], 1, -1 * this->lr);
    DElemArith(this->b[i], this->grads.db[i], 1, -1 * this->lr);
  }
}

void DataParallelNeuralNetwork::to_cpu(NeuralNetwork &nn)
{
  /**
   * TODO: Implement this function.
   * HINT: Use DeviceMatrix::to_cpu. You need to copy W[] and b[].
   */

  nn.W.resize(W.size());
  nn.b.resize(b.size());
  // move weights from gpu to cpu 
  for (size_t i = 0; i < nn.W.size(); i++) {
    assert(W[i].n_cols == nn.W[i].n_cols && W[i].n_rows == nn.W[i].n_rows);
    W[i].to_cpu(nn.W[i]); 
  }

  // move biases from gpu to cpu
  for (size_t i = 0; i < nn.b.size(); i++) {
    assert(b[i].n_cols == nn.b[i].n_cols && b[i].n_rows == nn.b[i].n_rows);
    b[i].to_cpu(nn.b[i]); 
  }

}