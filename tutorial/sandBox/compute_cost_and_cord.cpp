// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "tutorial/sandBox/compute_cost_and_cord.h"

void copmute_cost_and_cord_fun(
	const int e,
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& E,
	double& cost,
	Eigen::RowVectorXd& p, vector<MatrixXd>& Q_of_V)
{
	RowVectorXd place;
	Vector4d vectorCord;
	MatrixXd Q_Value, Q_ValueTemp;
	int vOneIndex = E(e, 0);
	int vTwoIndex = E(e, 1);
	Q_Value = Q_of_V.at(vOneIndex) + Q_of_V.at(vTwoIndex);
	Q_ValueTemp = Q_Value;
	vectorCord = findOptVValue( vOneIndex, vTwoIndex,V,Q_ValueTemp );
	cost = (vectorCord.transpose() * Q_Value * vectorCord)(0, 0);
	p(0) = vectorCord(0);
	p(1) = vectorCord(1);
	p(2) = vectorCord(2);
}
VectorXd findOptVValue(int vOneIndex, int vTwoIndex, const Eigen::MatrixXd& V, MatrixXd Q_Value)
{

	Vector4d ZeroVector(0, 0, 0, 1);
	Q_Value.row(3) = ZeroVector;


	if (Q_Value.determinant() != 0)
	{
		return (Q_Value.inverse() * ZeroVector);
	}
	return 0.5 * (V.row(vOneIndex) + V.row(vTwoIndex));

}