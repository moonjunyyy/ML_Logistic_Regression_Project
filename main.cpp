#pragma comment(lib, "../ML_Logistic_Regression_Project/x64/Release/Perceptron_Lib.lib")

#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <thread>
#include <algorithm>
#include "../Perceptron_Lib/perceptron.h"

using namespace std;

const int train_N	= 1000;  // number of training data
const int test_N	= 200;   // number of test data
const int nIn		= 2;        // dimensions of input data

int main(int argc, char* argv[])
{
	//
	// Declare (Prepare) variables and constants for perceptrons
	double	train_X[train_N][nIn];  // input data for training
	int		train_T[train_N];               // output data (label) for training

	double	test_X[test_N][nIn];  // input data for test
	int		test_T[test_N];               // label of inputs

	int		predicted_T[test_N];          // output data predicted by the model

	int		epochs = 2000;   // maximum training epochs
	double	learningRate = 1.;  // learning rate can be 1 in perceptrons

	//
	// Create training data and test data for demo.
	//
	// Let training data set for each class follow Normal (Gaussian) distribution here:
	//   class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
	//   class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
	//

	default_random_engine generator;
	normal_distribution<double> g1(-2.0, 1.5);
	normal_distribution<double> g2(2.0, 1.5);

	ofstream xxx("data.txt");
	// data set in class 1
	for (int i = 0; i < train_N / 2; i++) {
		train_X[i][0] = g1(generator);
		train_X[i][1] = g2(generator);
		train_T[i] = 1;

		xxx << train_X[i][0] << " " << train_X[i][1] << endl;
	}
	for (int i = 0; i < test_N / 2; i++) {
		test_X[i][0] = g1(generator);
		test_X[i][1] = g2(generator);
		test_T[i] = 1;
	}
	// data set in class 2
	for (int i = train_N / 2; i < train_N; i++) {
		train_X[i][0] = g2(generator);
		train_X[i][1] = g1(generator);
		train_T[i] = -1;
		xxx << train_X[i][0] << " " << train_X[i][1] << endl;
	}
	for (int i = test_N / 2; i < test_N; i++) {
		test_X[i][0] = g2(generator);
		test_X[i][1] = g1(generator);
		test_T[i] = -1;
	}

	xxx.close();

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
			classified_ += classifier.train(train_X[i], train_T[i], learningRate);
		}

		if (classified_ == train_N) break;  // when all data classified correctly

		epoch++;
		if (epoch > epochs) break;
	}
	std::cout << "iteration " << epoch << " of " << epochs << endl;
	std::cout << "w " << classifier.w[0] << " " << classifier.w[1] << endl;

	// test
	for (int i = 0; i < test_N; i++) {
		predicted_T[i] = classifier.predict(test_X[i]);
	}
	//
	// Evaluate the model
	//
	int		confusionMatrix[2][2];
	double	accuracy = 0.;
	double	precision = 0.;
	double	recall = 0.;
	// Java에서는 초기값이 0으로 되어있으나 C++에서는 정해주어야함
	confusionMatrix[0][0] = confusionMatrix[0][1] = confusionMatrix[1][0] = confusionMatrix[1][1] = 0;
	for (int i = 0; i < test_N; i++) {

		if (predicted_T[i] > 0) {
			if (test_T[i] > 0) {
				// True Positive
				accuracy += 1;
				precision += 1;
				recall += 1;
				confusionMatrix[0][0] += 1;
			}
			else {
				// False Positive
				confusionMatrix[1][0] += 1;
			}
		}
		else {
			if (test_T[i] > 0) {
				// False Negative
				confusionMatrix[0][1] += 1;
			}
			else {
				// True Negative
				accuracy += 1;
				confusionMatrix[1][1] += 1;
			}
		}

	}
	accuracy /= test_N;
	precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
	recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

	std::cout << "----------------------------" << endl;
	std::cout << "Perceptrons model evaluation" << endl;
	std::cout << "----------------------------" << endl;
	std::cout << "Accuracy:  " << accuracy * 100 << endl;
	std::cout << "Precision: " << precision * 100 << endl;
	std::cout << "Recall: " << recall * 100 << endl;
	getchar();
}