// This file is part of libigl, a simple c++ geometry processing library.
// 
// Copyright (C) 2016 Alec Jacobson <alecjacobson@gmail.com>
// 
// This Source Code Form is subject to the terms of the Mozilla Public License 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#include "shortest_edge_and_midpoint.h"

IGL_INLINE void igl::shortest_edge_and_midpoint(
  const int e,
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::MatrixXi & E,
  const Eigen::VectorXi & /*EMAP*/,
  const Eigen::MatrixXi & /*EF*/,
  const Eigen::MatrixXi & /*EI*/,
  double & cost,
  Eigen::RowVectorXd & p)
{

	/*
	RowVectorXd place;
	Vector4d vectorCord;
	MatrixXd Q_Value, Q_ValueTemp;
	Vector4d ZeroVector(0, 0, 0, 1);
	int vOneIndex = E(e, 0);
	int vTwoIndex = E(e, 1);
	Q_Value = Q_of_V.at(vOneIndex) + Q_of_V.at(vTwoIndex);
	Q_ValueTemp = Q_Value;
	Q_ValueTemp.row(3) = ZeroVector;
	if (Q_Value.determinant() != 0)
	{
		vectorCord = (Q_Value.inverse() * ZeroVector);
	}
	vectorCord = 0.5 * (V.row(vOneIndex) + V.row(vTwoIndex));
	cost = (vectorCord.transpose() * Q_Value * vectorCord)(0, 0);
	p(0) = vectorCord(0);
	p(1) = vectorCord(1);
	p(2) = vectorCord(2);
	*/
  cost = (V.row(E(e,0))-V.row(E(e,1))).norm();
  p = 0.5*(V.row(E(e,0))+V.row(E(e,1)));
}
