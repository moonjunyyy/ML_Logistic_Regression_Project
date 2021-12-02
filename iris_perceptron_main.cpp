/* 
   Author : Doug Young Suh
   Date :    2019 4 6

   Perceptron in Java -->  perceptron in C++

   input: iris_data.txt

   결과: Iris-setosa , Iris-virginica는 100%,   Iris-versicolor은 모두 한쪽으로만 판정...
   for  Iris-setosa T = 1, otherwise T = -1
*/
#include <iostream>
#include <fstream>
#include <random>
using namespace std;
#include "perceptron.h"
#define train_N  120  // number of training data
#define test_N   30   // number of test data
#define nIn  4        // dimensions of input data
// Declare (Prepare) variables and constants for perceptrons
double train_X[train_N][nIn];  // input data for training
int train_T[train_N];               // output data (label) for training

double test_X[test_N][nIn];  // input data for test
int test_T[test_N];               // label of inputs

void get_iris_data();
int * shuffle(int N);

void main() {
		//
		int predicted_T[test_N];          // output data predicted by the model
		int * shuffled = shuffle(train_N);
		int epochs = 2000;   // maximum training epochs
		double learningRate = 1.;  // learning rate can be 1 in perceptrons
		//
		// Create training data and test data for demo.
		//

		get_iris_data();
		//
		// Build SingleLayerNeuralNetworks model
		//
		int epoch = 0;  // training epochs
						// construct perceptrons
		Perceptrons classifier(nIn);

		// train models
		while (true) {
			int classified_ = 0;

			for (int i = 0; i < train_N; i++) {
				classified_ += classifier.train(train_X[shuffled[i]], train_T[shuffled[i]], learningRate);
			}

			if (classified_ == train_N) break;  // when all data classified correctly

			epoch++;
			if (epoch > epochs) break;
		}
		std::cout << "iteration " << epoch << " of " << epochs << endl; 
		std::cout << "w " << classifier.w[0] << " " << classifier.w[1] << " " << classifier.w[2] << " " << classifier.w[3] << endl;
		
		// test
		for (int i = 0; i < test_N; i++) {
			predicted_T[i] = classifier.predict(test_X[i]);
		}
		//
		// Evaluate the model
		//
		int confusionMatrix[2][2];
		double accuracy = 0.;
		double precision = 0.;
		double recall = 0.;
		// Java에서는 초기값이 0으로 되어있으나 C++에서는 정해주어야함
		confusionMatrix[0][0] = confusionMatrix[0][1] = confusionMatrix[1][0] = confusionMatrix[1][1] = 0;
		for (int i = 0; i < test_N; i++) {

			if (predicted_T[i] > 0) {
				if (test_T[i] > 0) {
					accuracy += 1;
					precision += 1;
					recall += 1;
					confusionMatrix[0][0] += 1;
				}
				else {
					confusionMatrix[1][0] += 1;
				}
			}
			else {
				if (test_T[i] > 0) {
					confusionMatrix[0][1] += 1;
				}
				else {
					accuracy += 1;
					confusionMatrix[1][1] += 1;
				}
			}
		}
		accuracy /= test_N;
		precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
		recall /= confusionMatrix[0][0] + confusionMatrix[0][1];


		std::cout << " confusion" << endl;
		std::cout << confusionMatrix[0][0] << " " << confusionMatrix[0][1] << " " << endl;
		std::cout << confusionMatrix[1][0] << " " << confusionMatrix[1][1] << " " << endl;
		
		std::cout << "----------------------------" << endl;
		std::cout << "Perceptrons model evaluation" << endl;
		std::cout << "----------------------------" << endl;
		std::cout << "Accuracy:  "<< accuracy * 100 << endl;
		std::cout << "Precision: "<< precision * 100 << endl;
		std::cout << "Recall: "<< recall * 100 << endl;
		getchar();
}

// read iris data
void get_iris_data() {
	ifstream iris_data("iris_data.txt");
	if (!iris_data) std::cout << " no input file \n";
	int it;
	int patterns = 3;
	for (int k = 0; k<patterns; k++) {
		int m = k * 40;
		for (int i = 0; i < 40; i++, m++) {
			iris_data >> train_X[m][0] >> train_X[m][1] >> train_X[m][2] >> train_X[m][3];
			iris_data >> it;  // it = 0, 1, or 2
			if (it == 2) train_T[m] = 1; else train_T[m] = -1;
		}
		m = k * 10;
		for (int i = 0; i < 10; i++, m++) {
			iris_data >> test_X[m][0] >> test_X[m][1] >> test_X[m][2] >> test_X[m][3];
			iris_data >> it;
			if (it == 2) test_T[m] = 1; else test_T[m] = -1;
		}
	}
}
// return an int array of random-shuffled positions
int * shuffle(int N) {
	int *aa = new int[N];
	int i, left, j;

	for (i = 0; i < N; i++) aa[i] = -1;
	for (i = 0; i < N; i++) {
		left = rand() % (N - i);
		for (j = 0; j < N; j++) if (aa[j] == -1) {
			if (left == 0) break; else left--;
		}
		aa[j] = i;
	}
	return aa;
}