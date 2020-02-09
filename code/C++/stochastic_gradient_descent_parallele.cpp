
#include <iostream>
#include <random>
#include <omp.h>


////////////////////   Autres fonctions  //////////////////////

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


float* average(float **arr, int n, int p)
{
    float* res = new float[p];
    for (int i=0; i<p; i++){
        res[i] = 0;
        for (int j=0; j<n; j++){
            res[i] += arr[i][j];
        }
        res[i] = res[i] / n;
    }

    return res;
}


float* average_para(float **arr, int n, int p, int num_threads)
{
	float* res = new float[p];
	
	omp_set_num_threads(num_threads);
	int id;
	
	#pragma omp parallel for private(id) shared(res, arr, n, p) 
	for (int i=0; i<p; i++){
		res[i]=0;
		for (int j=0; j<n; j++){
			res[i] += arr[i][j];
		}
		res[i] = res[i]/n;
	}
	return res;
}



//////////////////////////////// SGD P ///////////////////////////////////////////

float* stochastic_gradient_descent_parallel(float eta, float **X, float *y, int n, int p, int num_threads)
{
	float beta_t[p]; // = new float[p];
	float y_hat_t;
	float gradient_t[p]; // = new float[p];
	float** beta_hist;
	beta_hist = new float*[p];

	for (int i=0; i<(p); i++){
		beta_hist[i] = new float[n];
	}

	for (int k=0; k<p; k++){
		beta_t[k] = 0;
	}


	int* tab_index = new int[n];
	for (int i=0; i<n; i++){
		tab_index[i] = i;
	}
	shuffle(tab_index, n);

	omp_set_num_threads(num_threads);
	int id;
	
	//Dans la région parallèle on passe les tableaux beta_t, y_hat_t, gradient_t et eta en firstprivate car ces variables ont pour but d'être modifiées par les threads
	//On évite ainsi que les threads modifient les vecteurs beta/gradient des autres
	//Les variables p, tab_index ne sont pas modifiées par les threads.On peut donc les mettre en shared ou firsprivate. 
	//Shared évite de faire des copies des variables tandis que firstprivate améliorera la performance en facilitant l'accès des threads aux variables
	
	#pragma omp parallel for private(id) firstprivate(beta_t, y_hat_t, gradient_t, eta, p, tab_index) shared(beta_hist)  num_threads(num_threads) schedule(static)
	for (int j=0; j<n; j++){
	    eta = eta * 0.99999;
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
		beta_hist[k][j] = beta_t[k];
		}

	}
    
	return average_para(beta_hist, n, p, num_threads);
	//return average(beta_hist, n, p);

}

//////////////////////////////// SGD P Dynamic///////////////////////////////////////////

float* stochastic_gradient_descent_parallel_dynamic(float eta, float **X, float *y, int n, int p, int num_threads)
{
	float beta_t[p]; 
	float y_hat_t;
	float gradient_t[p]; 
	float** beta_hist;
	beta_hist = new float*[p];

	for (int i=0; i<(p); i++){
		beta_hist[i] = new float[n];
	}

	for (int k=0; k<p; k++){
		beta_t[k] = 0;
	}


	int* tab_index = new int[n];
	for (int i=0; i<n; i++){
		tab_index[i] = i;
	}
	shuffle(tab_index, n);

	omp_set_num_threads(num_threads);
	int id;
	
	//Dans la région parallèle on passe les tableaux beta_t, y_hat_t, gradient_t et eta en firstprivate car ces variables ont pour but d'être modifiées par les threads
	//On évite ainsi que les threads modifient les vecteurs beta/gradient des autres
	//Les variables p, tab_index ne sont pas modifiées par les threads.On peut donc les mettre en shared ou firsprivate. 
	//Shared évite de faire des copies des variables tandis que firstprivate améliorera la performance en facilitant l'accès des threads aux variables
	
	#pragma omp parallel for private(id) firstprivate(beta_t, y_hat_t, gradient_t, eta, p, tab_index) shared(beta_hist)  num_threads(num_threads) schedule(dynamic)
	for (int j=0; j<n; j++){
	     // Here the coeficient of decrease of eta might change depending on the number of data
            // It goes from 0.9999 (for n=10000) up to 0.999999 (for n=5000000 or more)
	    eta = eta * 0.9999;
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
		beta_hist[k][j] = beta_t[k];
		}

	}
    
	return average_para(beta_hist, n, p, num_threads);
	//return average(beta_hist, n, p);
}


int main()
{




////////////////////   Generation des donnÃ©es   //////////////////////

    const int n = 1000000;
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
    float eta = 0.00055;
    float* beta = stochastic_gradient_descent_parallel(eta, X, y, n, p, 2);
    //float* beta = stochastic_gradient_descent_parallel_dynamic(eta, X, y, n, p, 4);

    for (int j=0; j<p; j++){
        std::cout << beta[j] << std::endl;
    }

    std::cout << "---------------------------------" << std::endl;

    for (int j=0; j<p; j++){
        std::cout << coefs[j] << std::endl;
    }
    float eq = 0;
    for (int j=0; j<p; j++){
		eq += pow((beta[j]-coefs[j]),2);
    }
    std::cout<<"---------------EQ----------------"<<std::endl;
    std::cout<<eq<<std::endl;
    return 0;
}
