
#include <iostream>
#include <random>
#include <cmath>

////////////////////   Descente de gradient   //////////////////////

void shuffle(int *arr, int n)
{
    for (int i = 0; i < n - 1; i++) 
    {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = arr[j];
        arr[j] = arr[i];
        arr[i] = t;
    }
}


float* stochastic_gradient_descent(float eta, int s_iter, float **X, float *y, int n, int p)
{
    float* beta_t = new float[p];
    float y_hat_t;
    float* gradient_t = new float[p];
    float** beta_hist;
    beta_hist = new float*[n*s_iter];
    for (int i=0; i<(n*s_iter); i++){
        beta_hist[i] = new float[p];
    }

    for (int k=0; k<p; k++){
        beta_t[k] = 0;
    }

    int* tab_index = new int[n];
    for (int i=0; i<n; i++){
        tab_index[i] = i;
    }
    shuffle(tab_index, n);

    for (int s=0; s<s_iter; s++){
        int j = 0;
        while (j<n){
            // Here the coeficient of decrease of eta might change depending on the number of data
            // It goes from 0.9999 (for n=10000) up to 0.999999 (for n=5000000 or more)
            eta = eta * 0.999999;
            float* X_j = X[tab_index[j]];
            float y_j = y[tab_index[j]];
            y_hat_t = 0;
            for (int k=0; k<p; k++){
                y_hat_t += X_j[k] * beta_t[k];
            }
            for (int k=0; k<p; k++){
                gradient_t[k] = 0;
                gradient_t[k] = (X_j[k] * (y_j - y_hat_t)) * (-2.0);
                beta_t[k] = beta_t[k] - eta * gradient_t[k];
            }
            j += 1;
        }
    }
    return beta_t;
}


int main()
{

////////////////////   Generation des données   //////////////////////

    const int n = 500000;
    const int p = 8;

    std::random_device rd;
    std::mt19937 gen(rd());

    float* coefs;
    coefs = new float[p];

    for (int i=0; i<p; i++){
        std::normal_distribution<float> d(0.0, 1.0);
        coefs[i] = d(gen) * 3.0;
    }

    float** X;
    X = new float*[n];
    for (int i=0; i<n; i++){
        X[i] = new float[p];
    }

    for (int i=0; i<n; i++){
        for (int j=0; j<p; j++){
            if (j == 0){
                X[i][j] = 1;
            }
            else{
                std::uniform_real_distribution<float> d(0, 10);
                X[i][j] = d(gen);
            }
        }
    }

    float* y;
    y = new float[n];
    for (int i=0; i<n; i++){
        y[i] = 0;
        for (int j=0; j<p; j++){
            std::normal_distribution<float> d(0.0, 1.0);
            float epsilon = d(gen) * 2;
            y[i] = y[i] + coefs[j] * X[i][j] + epsilon;
        }
    }

////////////////////   Descente de gradient stochastique   //////////////////////

    int s_iter = 1;
    float eta = 0.00005;

    float* beta = stochastic_gradient_descent(eta, s_iter, X, y, n, p);

    for (int j=0; j<p; j++){
        std::cout << beta[j] << std::endl;
    }

    std::cout << "---------------------------------" << std::endl;

    for (int j=0; j<p; j++){
        std::cout << coefs[j] << std::endl;
    }

    std::cout << "---------------------------------" << std::endl;

    float sum_square = 0;
    for (int i=0; i<p; i++){
        sum_square += pow((coefs[i] - beta[i]),2);
    }

    std::cout << sum_square << std::endl;

    return 0;
}
