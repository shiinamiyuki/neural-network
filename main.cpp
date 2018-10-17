#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
template<typename T>
struct Mat {
    std::vector<T> data;
    int M;
    int N;
    bool transposed;
    Mat(){}
    Mat(int m, int n) {
        M = m;
        N = n;
        data.resize(m * n);
        for(auto & i:data){
           i =  (double) rand() / RAND_MAX;
        }
        transposed = false;
    }

    T &operator()(int i, int j) {
        if (!transposed) {
            //  assert(i * N + j < data.size());
            return data[i * N + j];
        } else {
            //  assert(i + M * j < data.size());
            return data[i + M * j];
        }
    }

    const T &operator()(int i, int j) const {
        if (!transposed) {
            return data[i * N + j];
        } else {
            return data[i + M * j];
        }
    }

    Mat<T> &operator+=(const Mat<T> &rhs) {
        assert(M == rhs.M && N == rhs.N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                (*this)(i, j) += rhs(i, j);
            }
        }
        return *this;
    }

    Mat<T> &operator-=(const Mat<T> &rhs) {
        assert(M == rhs.M && N == rhs.N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                (*this)(i, j) -= rhs(i, j);
            }
        }
        return *this;
    }

    Mat<T> operator-(const Mat<T> &rhs) {
        decltype(*this) m = *this;
        m -= rhs;
        return m;
    }

    Mat<T> operator+(const Mat<T> &rhs) {
        decltype(*this) m = *this;
        m += rhs;
        return m;
    }

    Mat<T> operator*(const Mat<T> &rhs) {
        assert(N == rhs.M);
        Mat<T> c(M, rhs.N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < rhs.N; j++) {
                c(i, j) = 0;
                for (int k = 0; k < N; k++) {
                    c(i, j) += (*this)(i, k) * rhs(k, j);
                }
            }
        }
        return c;
    }
    Mat<T> operator*(const T scale) {
        Mat<T> c(M, N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                c(i, j) =(*this)(i,j) * scale;
            }
        }
        return c;
    }
    Mat<T> dot(const Mat<T>&rhs){
        assert(M == rhs.M && N == rhs.N);
        Mat<T> m(M,N);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                m(i,j)  = (*this)(i,j) * rhs(i,j);
            }
        }
        return m;
    }
    void transpose() {
        transposed = !transposed;
        std::swap(M, N);
    }
};
template <typename  T>
void sigmoid(const Mat<T>&in,Mat<T>& out){
    assert(in.M == out.M && in.N == out.N);
    for(int i =0 ;i<in.M;i++)
        for(int j = 0;j<in.N;j++){
            out(i,j) = 1/(exp(-in(i,j))+1);
        }
}
template <typename  T>
void derivative(const Mat<T>&in,Mat<T>& out){
    assert(in.M == out.M && in.N == out.N);
    for(int i =0 ;i<in.M;i++)
        for(int j = 0;j<in.N;j++){
            out(i,j) = (1-in(i,j)) * in(i,j);
        }
}

struct NN {
    Mat<double> in, w1, b1, hidden, w2, b2, out;
    Mat<double> e1,e2;
    NN(int a, int b, int c) :
    in(1, a), w1(a, b), b1(1, b), hidden(1, b), w2(b, c), b2(1, c), out(1, c),
    e2(1,c),e1(1,b){
        for (auto &i: w1.data) {
            i = (double) rand() / RAND_MAX;
        }
        for (auto &i: w2.data) {
            i = (double) rand() / RAND_MAX;
        }
    }

    void feedForward(const std::vector<double> &input) {
        assert(input.size() == in.N);
        for (int i = 0; i < in.N; i++) {
            in(0, i) = input[i];
        }
        sigmoid(in * w1 + b1 ,hidden);
        sigmoid(hidden * w2 +b2,out);
    }

    void backPropagate(const std::vector<double>& target,const double rate =0.001) {
        assert(target.size() == out.N);
        double loss = 0;
        static int count = 0;
        for(int i =0 ;i<out.N;i++){
            double x = out(0,i)-target[i];
            loss += x*x;
            e2(0,i) = x * (1 - out(0,i))* out(0,i);
        }
        loss *= 0.5;
        if(count ++ % 1000 == 0)
            printf("%d  %lf\n",count,loss);
        hidden.transpose();
        w2 += hidden * e2 * -rate;
        b2 += e2 * -rate;
        hidden.transpose();
        w2.transpose();
        e1 = e2 * w2;
        w2.transpose();
        for(int i = 0;i<hidden.N;i++){
            e1(0,i) *= (1 - hidden(0,i))*hidden(0,i);
        }
        in.transpose();
        w1 += in * e1 * -rate;
        b1 += e1 * -rate;
        in.transpose();
    }

};
struct DNN{
    std::vector<Mat<double>> layers;
    std::vector<Mat<double>> errors;
    std::vector<Mat<double>> weights;
    std::vector<Mat<double>> bias;
    int count;
    DNN(const std::vector<int> dim){
        count = 0;
        layers.emplace_back(Mat<double>(1,dim[0]));
        bias.emplace_back(Mat<double>());
        weights.emplace_back(Mat<double>());
        errors.emplace_back(Mat<double>());
        for(int i = 1;i<dim.size();i++){
            layers.emplace_back(Mat<double>(1,dim[i]));
            errors.emplace_back(Mat<double>(1,dim[i]));
            bias.emplace_back(Mat<double>(1,dim[i]));
            weights.emplace_back(Mat<double>(dim[i-1],dim[i]));
        }
    }
    void feedForward(const std::vector<double> &input){
        for(int i =0 ;i<input.size();i++){
            layers[0](0,i) = input[i];
        }
        for(int i = 1 ;i<layers.size();i++){
            sigmoid(layers[i-1] * weights[i] + bias[i], layers[i]);
        }
    }
    void backPropagate(const std::vector<double>& target,const double rate =0.001){
        double loss = 0;
        for(int i =0 ;i<layers.back().N;i++){
            double x = layers.back()(0,i)-target[i];
            loss += x*x;
            errors[errors.size() - 1](0,i) = x * (1 - layers.back()(0,i))* layers.back()(0,i);
        }
        if(count ++ % 1000 == 0)
            printf("%d  %lf\n",count,loss);
        for(auto i = layers.size() - 1;i>=1;i--){
            layers[i - 1].transpose();
            weights[i] += layers[i - 1] * errors[i] * -rate;
            bias[i] += errors[i] * -rate;
            layers[i -1].transpose();
            weights[i].transpose();
            errors[i-1] = errors[i]   * weights[i];
            for(int k = 0;k<errors[i-1].N;k++){
                errors[i-1](0,k) *= (1 - layers[i - 1](0,k))*layers[i - 1](0,k);
            }
            weights[i].transpose();
        }
    }
};
int main() {
    DNN nn({2, 10, 1});
    std::pair<std::vector<double>,std::vector<double>> data[] {{{0,1},{1}},
     {{1,0},{1}},
     {{1,1},{0}},{{0,0},{0}}};
    for(int iter = 0;iter<40000;iter++) {
        for (int i = 0; i < 4; i++) {
            nn.feedForward(data[i].first);
            nn.backPropagate(data[i].second,2);
        }
    }
    for(int i = 0;i<4;i++) {
        nn.feedForward(data[i].first);
        for (auto i:nn.layers.back().data) {
            printf("%lf ", i);
        }
    }
    return 0;
}