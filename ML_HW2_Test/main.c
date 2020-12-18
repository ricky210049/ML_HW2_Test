#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double);//sigmoid
double sigmoid_diff(double);//sigmoid對z做微分
double** arr(int, int);//創建二維陣列
double** transpose(double**, int, int);//轉置矩陣
double** rnd(int, int);//產生隨機二維陣列

int main() {

	//使用者參數
	const char filename[] = "data/579_train.csv"; //training資料集位置
	const char filename_test[] = "data/579_test.csv"; //testing資料集位置
	const int layer1 =7;//第一層神經元個數
	const int layer2 = 7;//第二層神經元個數
	const int outputLayer = 3;//輸出層神經元個數
	const double lr = 0.01;//學習率
	const int iteration = 1000;//迭代次數

	//----------Training------------//

	int row, col;
	char* line[1024];
	row = get_row(filename);
	col = get_col(filename);
	
	//儲存讀入檔案資料
	double** a = arr(row, col);

	//賦值
	FILE* stream = fopen(filename, "r");
	int m = 0;
	while (fgets(line, 1024, stream)) {
		char* token = strtok(line, ",");
		int n = 0;
		while (token) {

			a[m][n] = atof(token);
			token = strtok(NULL, ",");
			n++;

		}
		m++;
	}

	//trainY
	double* train_y = (double*)malloc(row * sizeof(double));
	for (int i = 0; i < row; i++) {
		train_y[i] = a[i][2];
	}

	//trainX
	double** train_x = arr(row, col - 1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col - 1; j++) {
			train_x[i][j] = a[i][j];
		}
	}

	//y head資料處理
	double** y_head = arr(row, outputLayer);
	for (int i = 0; i < row; i++) {
		if (train_y[i] == 1) {
			y_head[i][0] = 1;
			y_head[i][1] = 0;
			y_head[i][2] = 0;
		}
		else if (train_y[i] == 2) {
			y_head[i][0] = 0;
			y_head[i][1] = 1;
			y_head[i][2] = 0;
		}
		else if (train_y[i] == 3) {
			y_head[i][0] = 0;
			y_head[i][1] = 0;
			y_head[i][2] = 1;
		}
	}

	//呼叫亂數

	//呼叫亂數w1(輸入層到第一層隱藏層的鍵結值 範圍:-1~1)
	double** w1 = rnd(layer1, col - 1);

	//呼叫亂數b1
	double* b1 = (double*)malloc(layer1 * sizeof(double));
	for (int i = 0; i < layer1; i++) {
		int rnd1 = (rand() % 201) + 1 - 100;
		double rnd2 = (double)rnd1 * 0.01;
		b1[i] = rnd2;
	}

	//呼叫亂數w2(輸入層到第一層隱藏層的鍵結值 範圍:-1~1)
	double** w2 = rnd(layer2, layer1);

	//呼叫亂數b2
	double* b2 = (double*)malloc(layer2 * sizeof(double));
	for (int i = 0; i < layer2; i++) {
		int rnd1 = (rand() % 201) + 1 - 100;
		double rnd2 = (double)rnd1 * 0.01;
		b2[i] = rnd2;
	}

	//呼叫亂數w_final(隱藏層到輸出層的鍵結值 範圍:-1~1)
	double** w_final = rnd(outputLayer, layer1);


	//宣告
	//存放第一層神經元的值 z
	double** sum1 = arr(row, layer1);

	//存放第一層神經元的值(活化過後的) a
	double** sum1_a = arr(row, layer1);

	//存放第二層神經元的值 z
	double** sum2 = arr(row, layer2);

	//存放第二層神經元的值(活化過後的) a
	double** sum2_a = arr(row, layer2);

	//存放輸出層的值 z
	double** sum_final = arr(row, outputLayer);

	//存放softmax的值
	double** softmax = arr(row, outputLayer);

	//每次的loss function (cross entropy) L
	double* c = (double*)malloc(row * sizeof(double));

	//final loss 對z做微分 
	double** loss_diff = arr(row, outputLayer);

	//wfinal亂數矩陣轉置
	double** wfinal_tran = transpose(w_final, layer2, outputLayer);

	//第一層神經元做L對z的微分
	double** layer1_diff = arr(row, layer1);

	//第一層神經元做L對z的微分
	double sum3 = 0.0;

	//第二層神經元做L對z的微分
	double** layer2_diff = arr(row, layer2);

	//w1亂數矩陣轉置
	double** w1_tran = transpose(w1, col - 1, layer1);

	//w2亂數矩陣轉置
	double** w2_tran = transpose(w2, layer1, layer2);

	//輸入層到第一層的L對w的微分總和
	double** w_layer1_diff = arr(layer1, col - 1);

	//第一層到第二層的L對w微分總和
	double** w_layer2_diff = arr(layer2, layer1);
	
	//第二層層到輸出層的L對w微分總和
	double** w_outputlayer_diff = arr(outputLayer, layer2);
	
	//第一層L對b的微分總和
	double* b1_sum = (double*)malloc(layer1 * sizeof(double));
	
	//第二層L對b的微分總和
	double* b2_sum = (double*)malloc(layer2 * sizeof(double));

	//total_loss
	double Loss = 0;

	//Training開始**************************************************************************************************************************************//

	for (int iii = 0; iii < iteration; iii++) {

		//迭代開始前初始化
		Loss = 0;
		for (int i = 0; i < layer1; i++) for (int j = 0; j < col - 1; j++) w_layer1_diff[i][j] = 0;
		for (int i = 0; i < layer2; i++) for (int j = 0; j < layer1; j++) w_layer2_diff[i][j] = 0;
		for (int i = 0; i < outputLayer; i++) for (int j = 0; j < layer2; j++)w_outputlayer_diff[i][j] = 0;
		for (int i = 0; i < layer1; i++)b1_sum[i] = 0;
		for (int i = 0; i < layer2; i++)b2_sum[i] = 0;

		//-------------------------------------------------------------------------------------//
		for (int ii = 0; ii < row; ii++) {

			//計算第一層神經元的值 z
			double wx1 = 0;
			for (int i = 0; i < layer1; i++) {

				wx1 = 0;
				for (int j = 0; j < col - 1; j++) {

					wx1 += w1[i][j] * train_x[ii][j];

				}
				sum1[ii][i] = wx1+b1[i];
			}

			//計算第一層神經元的值(活化過後的) a
			for (int i = 0; i < layer1; i++) {

				sum1_a[ii][i] = sigmoid(sum1[ii][i]);

			}

			//計算第二層神經元的值 z
			double wx2 = 0;
			for (int i = 0; i < layer2; i++) {

				wx2 = 0;
				for (int j = 0; j < layer1; j++) {

					wx2 += w2[i][j] * sum1_a[ii][j];

				}
				sum2[ii][i] = wx2+b2[i];
				
			}

			//計算第二層神經元的值(活化過後的) a
			for (int i = 0; i < layer2; i++) {

				sum2_a[ii][i] = sigmoid(sum2[ii][i]);

			}

			//計算輸出層神經元的值 z
			double wx_final = 0;
			for (int i = 0; i < outputLayer; i++) {

				wx_final = 0;
				for (int j = 0; j < layer2; j++) {

					wx_final += w_final[i][j] * sum2_a[ii][j];

				}
				sum_final[ii][i] = wx_final;

			}

			//softmax
			double wx_softmax = 0;
			for (int i = 0; i < outputLayer; i++) {
				wx_softmax += exp(sum_final[ii][i]);
			}

			for (int j = 0; j < outputLayer; j++) {
				softmax[ii][j] = exp(sum_final[ii][j]) / wx_softmax;
			}

			//Back propagation

			//final loss 對z做微分 
			for (int i = 0; i < outputLayer; i++) {
				loss_diff[ii][i] = softmax[ii][i] - y_head[ii][i];
			}
			//printf("\n");

			//第二層神經元做L對z的微分
			for (int i = 0; i < layer2; i++) {
				double sum = 0.0;
				for (int j = 0; j < outputLayer; j++) {
					sum += loss_diff[ii][j] * wfinal_tran[i][j];
				}
				layer2_diff[ii][i] = sum * sigmoid_diff(sum2[ii][i]);
				//printf("\n%f ", layer1_diff[ii][i]);
			}

			//第一層神經元做L對z的微分
			for (int i = 0; i < layer1; i++) {
				double sum = 0.0;
				for (int j = 0; j < layer2; j++) {
					sum += layer2_diff[ii][j] * w2_tran[i][j];
				}
				layer1_diff[ii][i] = sum * sigmoid_diff(sum1[ii][i]);
			}


			for (int i = 0; i < outputLayer; i++) {
				for (int j = 0; j < layer2; j++) {
					w_outputlayer_diff[i][j] += sum2_a[ii][j] * loss_diff[ii][i];
				}
			}

			for (int i = 0; i < layer2; i++) {
				for (int j = 0; j < layer1; j++) {
					w_layer2_diff[i][j] += sum1_a[ii][j] * layer2_diff[ii][i];
				}
			}

			for (int i = 0; i < layer1; i++) {
				for (int j = 0; j < col - 1; j++) {
					w_layer1_diff[i][j] += train_x[ii][j] * layer1_diff[ii][i];
				}
			}

			for (int i = 0; i < layer1; i++) {
				b1_sum[i] += layer1_diff[ii][i];
			}

			for (int i = 0; i < layer1; i++) {
				b2_sum[i] += layer2_diff[ii][i];
			}

			//每次的loss function (cross entropy) L
			c[ii] = 0;
			for (int i = 0; i < outputLayer; i++) {
				c[ii] -= y_head[ii][i] * log(softmax[ii][i]);
			}
			Loss += c[ii];

		}

		//-----------------------------------------------------------------------------------------//

		printf("\n%f ", Loss);


		//更新權重
		for (int i = 0; i < outputLayer; i++) {
			for (int j = 0; j < layer2; j++) {
				w_final[i][j] = w_final[i][j] - lr * w_outputlayer_diff[i][j];
			}
		}

		for (int i = 0; i < layer2; i++) {
			for (int j = 0; j < layer1; j++) {
				w2[i][j] = w2[i][j] - lr * w_layer2_diff[i][j];
			}
		}

		for (int i = 0; i < layer1; i++) {
			for (int j = 0; j < col - 1; j++) {
				w1[i][j] = w1[i][j] - lr * w_layer1_diff[i][j];
			}
		}

		for (int i = 0; i < layer1; i++) {
			b1[i] = b1[i] - lr * b1_sum[i];
		}

		for (int i = 0; i < layer1; i++) {
			b2[i] = b2[i] - lr * b2_sum[i];
		}

	}
	printf("\n\nTotal Training Loss: %f ", Loss);
	//************************************************************************************************************************************************//

	//印出調整後的weight與bias
	printf("\n\nw1\n");
	for (int i = 0; i < layer1; i++) {
		for (int j = 0; j < col - 1; j++) {
			printf("%f ",w1[i][j]);
		}
		printf("\n");
	}

	printf("\nw2\n");
	for (int i = 0; i < layer2; i++) {
		for (int j = 0; j < layer1; j++) {
			printf("%f ", w2[i][j]);
		}
		printf("\n");
	}

	printf("\nwfinal\n");
	for (int i = 0; i <outputLayer; i++) {
		for (int j = 0; j < layer2; j++) {
			printf("%f ", w_final[i][j]);
		}
		printf("\n");
	}

	printf("\nb1\n");
	for (int i = 0; i < layer1; i++) {
		printf("%f\n",b1[i]);
	}

	printf("\nb2\n");
	for (int i = 0; i < layer2; i++) {
		printf("%f\n", b2[i]);
	}
	printf("\n");



	//-----------Testing----------//
	int row_test, col_test;
	double testLoss = 0;
	row_test = get_row(filename_test);
	col_test = get_col(filename_test);

	//每次的loss function (cross entropy) L
	double* c_test = (double*)malloc(row_test * sizeof(double));

	//儲存讀入檔案資料
	double** a1 = arr(row_test, col_test);

	//賦值
	FILE* stream1 = fopen(filename_test, "r");
	int m1 = 0;
	while (fgets(line, 1024, stream1)) {
		char* token = strtok(line, ",");
		int n = 0;
		while (token) {

			a1[m1][n] = atof(token);
			//printf("%f ", a[m][n]);
			token = strtok(NULL, ",");
			n++;

		}
		//printf("\n");
		m1++;
	}

	//testX
	double** test_x = arr(row_test, col_test - 1);
	for (int i = 0; i < row_test; i++) {
		for (int j = 0; j < col_test - 1; j++) {
			test_x[i][j] = a1[i][j];
			//printf("%f ", train_x[i][j]);
		}
		//printf("\n");
	}

	//testY
	double* test_y = (double*)malloc(row * sizeof(double));
	for (int i = 0; i < row_test; i++) {
		test_y[i] = a1[i][2];
	}

	//test y head資料處理
	double** y_head_test = arr(row_test, outputLayer);
	for (int i = 0; i < row_test; i++) {
		if (test_y[i] == 1) {
			y_head_test[i][0] = 1;
			y_head_test[i][1] = 0;
			y_head_test[i][2] = 0;
		}
		else if (test_y[i] == 2) {
			y_head_test[i][0] = 0;
			y_head_test[i][1] = 1;
			y_head_test[i][2] = 0;
		}
		else if (test_y[i] == 3) {
			y_head_test[i][0] = 0;
			y_head_test[i][1] = 0;
			y_head_test[i][2] = 1;
		}
	}


	double accuracy = 0;
	
	for (int ii = 0; ii < row_test; ii++) {

		//計算第一層神經元的值(活化過後的) a
		double wx1 = 0;
		for (int i = 0; i < layer1; i++) {

			wx1 = 0;
			for (int j = 0; j < col_test - 1; j++) {

				wx1 += w1[i][j] * test_x[ii][j];

			}
			sum1_a[ii][i] = sigmoid( wx1 + b1[i] );
			//printf("%f\n", sum1[ii][i]);
		}

	

		//計算第二層神經元的值(活化過後的) a
		double wx2 = 0;
		for (int i = 0; i < layer2; i++) {

			wx2 = 0;
			for (int j = 0; j < layer1; j++) {

				wx2 += w2[i][j] * sum1_a[ii][j];

			}
			sum2_a[ii][i] = sigmoid( wx2 + b2[i] );
			//printf("%f\n", sum1[ii][i]);
		}


		//計算輸出層神經元的值 z
		double wx_final = 0;
		for (int i = 0; i < outputLayer; i++) {

			wx_final = 0;
			for (int j = 0; j < layer2; j++) {

				wx_final += w_final[i][j] * sum2_a[ii][j];

			}
			sum_final[ii][i] = wx_final;
			//printf(" %f", sum_final[ii][i]);
		}

		//softmax
		double wx_softmax = 0;
		for (int i = 0; i < outputLayer; i++) {
			wx_softmax += exp(sum_final[ii][i]);
		}

		for (int j = 0; j < outputLayer; j++) {
			softmax[ii][j] = exp(sum_final[ii][j]) / wx_softmax;
			//printf(" %d ", (int)(softmax[ii][j]+0.5));//四捨五入
			
		}
		//printf("\n");

		if ((int)(softmax[ii][0] + 0.5) == 1) {
			printf("預測結果: 1\n"); accuracy++;
		}
		else if ((int)(softmax[ii][1] + 0.5) == 1) {
			printf("預測結果: 2\n"); accuracy++;
		}
		else if ((int)(softmax[ii][2] + 0.5) == 1) {
			printf("預測結果: 3\n"); accuracy++;
		}

		for (int i = 0; i < outputLayer; i++)
		{
			if ((int)(softmax[ii][i] + 0.5) == test_y[ii]) {
				
			}
		}

		//每次的loss function (cross entropy) L
		c_test[ii] = 0;
		for (int i = 0; i < outputLayer; i++) {
			c_test[ii] -= y_head_test[ii][i] * log(softmax[ii][i]);
		}
		testLoss += c_test[ii];

	}
	printf("\nTotal Training Loss: %f ", Loss);
	printf("\nTotal Testing Loss: %f\n", testLoss);
	printf("\n準確率: %f\n", accuracy / row_test );




	return 0;

}

//sigmoid
double sigmoid(double z) {
	double a;
	a = 1 / (1 + exp(z));
	return a;
}

//sigmoid對z做微分
double sigmoid_diff(double z) {
	double a, b;
	a = 1 / (1 + exp(z));
	b = a * (1 - a);
	return b;
}

//轉置矩陣
double** transpose(double** matrix, int row, int col) {
	double** tran = arr(row, col);
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++) {
			tran[j][i] = matrix[i][j];
		}
	}
	return tran;
}


//創建二維陣列
double** arr(int row, int col) {
	double** a = (double**)malloc(row * sizeof(double*));
	for (int i = 0; i < row; i++) {
		a[i] = (double*)malloc(col * sizeof(double));
	}
	return a;
}

//呼叫-1~1二維亂數
double** rnd(int row, int col) {
	double** w = arr(row, col);
	srand(time(0));
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			int rnd1 = (rand() % 201) + 1 - 100;
			double rnd2 = (double)rnd1 * 0.01;
			w[i][j] = rnd2;
		}
	}
	return w;
}



int get_row(char* filename) {

	char* line[1024];//1024是一行最大的大小
	int i = 0;
	FILE* stream = fopen(filename, "r");
	while (fgets(line, 1024, stream)) {
		i++;
	}
	fclose(stream);
	return i;
}

int get_col(char* filename) {

	char line[1024];
	int i = 0;
	FILE* stream = fopen(filename, "r");
	fgets(line, 1024, stream);  //第一行
	char* token = strtok(line, ",");
	while (token) {
		token = strtok(NULL, ",");
		i++;
	}
	fclose(stream);
	return i;
}