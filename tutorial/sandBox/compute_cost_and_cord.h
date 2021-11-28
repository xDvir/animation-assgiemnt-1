#pragma once

#include <Eigen/Core>


#include <vector>
#include <string>
using namespace std;
using namespace Eigen;
using namespace igl;

class compute_cost_and_cord
{

public:
	void copmute_cost_and_cord_fun(
		const int e,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& E,
		double& cost,
		Eigen::RowVectorXd& p, vector<MatrixXd>& Q_of_V);
	VectorXd findOptVValue(int vOneIndex, int vTwoIndex, const Eigen::MatrixXd& V, MatrixXd Q_Value);

private:
	// Prepare array-based edge data structures and priority queue


};

#ifndef IGL_STATIC_LIBRARY
#  include "compute_cost_and_cord.cpp"
#endif