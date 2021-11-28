// This file is part of libigl, a simple c++ geometry processing library.
//
// Copyright (C) 2014 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "Viewer.h"

//#include <chrono>
#include <thread>

#include <Eigen/LU>


#include <cmath>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <cassert>

#include <igl/project.h>
//#include <igl/get_seconds.h>
#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/adjacency_list.h>

#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/massmatrix.h>
#include <igl/file_dialog_open.h>
#include <igl/file_dialog_save.h>
#include <igl/quat_mult.h>
#include <igl/axis_angle_to_quat.h>
#include <igl/trackball.h>
#include <igl/two_axis_valuator_fixed_up.h>
#include <igl/snap_to_canonical_view_quat.h>
#include <igl/unproject.h>
#include <igl/serialize.h>
#include <igl/edge_flaps.h>
#include <igl/parallel_for.h>
#include <igl/read_triangle_mesh.h>

#include <igl/shortest_edge_and_midpoint.h>
#include "igl/collapse_edge.h"
using namespace std;
using namespace Eigen;
using namespace igl;
MatrixXd V;
MatrixXi F;
MatrixXd C;
MatrixXi  E;
VectorXi EMAP;
VectorXi EQ;
igl::min_heap< std::tuple<double, int, int>>QHeap;
vector<MatrixXd> Q_of_V;
MatrixXi EI;
vector<set<int>> VF;
MatrixXi EF;

// Internal global variables used for glfw event handling
//static igl::opengl::glfw::Viewer * __viewer;
static double highdpi = 1;
static double scroll_x = 0;
static double scroll_y = 0;


int num_collapsed;
namespace igl
{
	namespace opengl
	{
		namespace glfw
		{

			void Viewer::Init(const std::string config)
			{


			}

			IGL_INLINE Viewer::Viewer() :
				data_list(1),
				selected_data_index(0),
				next_data_id(1),
				isPicked(false),
				isActive(false)
			{
				data_list.front().id = 0;



				// Temporary variables initialization
			   // down = false;
			  //  hack_never_moved = true;
				scroll_position = 0.0f;

				// Per face
				data().set_face_based(false);


#ifndef IGL_VIEWER_VIEWER_QUIET
				const std::string usage(R"(igl::opengl::glfw::Viewer usage:
  [drag]  Rotate scene
  A,a     Toggle animation (tight draw loop)
  F,f     Toggle face based
  I,i     Toggle invert normals
  L,l     Toggle wireframe
  O,o     Toggle orthographic/perspective projection
  T,t     Toggle filled faces
  [,]     Toggle between cameras
  1,2     Toggle between models
  ;       Toggle vertex labels
  :       Toggle face labels)"
				);
				std::cout << usage << std::endl;
#endif
			}

			IGL_INLINE Viewer::~Viewer()
			{
			}

			IGL_INLINE bool Viewer::load_mesh_from_file(
				const std::string& mesh_file_name_string)
			{

				// Create new data slot and set to selected
				if (!(data().F.rows() == 0 && data().V.rows() == 0))
				{
					append_mesh();
				}
				data().clear();

				size_t last_dot = mesh_file_name_string.rfind('.');
				if (last_dot == std::string::npos)
				{
					std::cerr << "Error: No file extension found in " <<
						mesh_file_name_string << std::endl;
					return false;
				}

				std::string extension = mesh_file_name_string.substr(last_dot + 1);

				if (extension == "off" || extension == "OFF")
				{
					Eigen::MatrixXd V;
					Eigen::MatrixXi F;
					vector<set<int>>* VF = new vector<set<int>>;
					if (!igl::readOFF(mesh_file_name_string, V, F, VF))
						return false;

					data().set_mesh(V, F);
				}
				else if (extension == "obj" || extension == "OBJ")
				{
					Eigen::MatrixXd corner_normals;
					Eigen::MatrixXi fNormIndices;
					Eigen::MatrixXd UV_V;
					Eigen::MatrixXi UV_F;
					Eigen::MatrixXd V;
					Eigen::MatrixXi F;

					if (!(
						igl::readOBJ(
							mesh_file_name_string,
							V, UV_V, corner_normals, F, UV_F, fNormIndices)))
					{
						return false;
					}

					data().set_mesh(V, F);
			
					//yoson
					if (UV_V.rows() > 0)
					{
						data().set_uv(UV_V, UV_F);
					}

				}
				else
				{
					// unrecognized file type
					printf("Error: %s is not a recognized file type.\n", extension.c_str());
					return false;
				}

				data().compute_normals();
				data().uniform_colors(Eigen::Vector3d(51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0),
					Eigen::Vector3d(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0),
					Eigen::Vector3d(255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0));

				// Alec: why?
				if (data().V_uv.rows() == 0)
				{
					data().grid_texture();
				}


				//for (unsigned int i = 0; i<plugins.size(); ++i)
				//  if (plugins[i]->post_load())
				//    return true;

				/*
				our: here we build 2 data structures that will be used in the future:
				1. vector of dataVectors whicj eche one of them belongs for a specific 
				   object that has been loaded. 
					this dataVector is a Struct that holds all the information regarding	
					this object
				2.  vector of sets, while each set at index i belongs to the vertex i in 
					this vector.
					this set will hold all of the faces that touches the vertex.
																
				
				*/
				vector<set<int>> OVtemp;
				dataVector* shape = new dataVector();
				shape->OV = data().V;
				shape->OF = data().F;
				shape->V = data().V;
				shape->F = data().F;
				OVtemp.resize(data().V.rows());
				for (size_t i = 0; i < data().F.rows(); i++)
				{

					for (size_t j = 0; j < data().F.row(i).cols(); j++)
					{
						OVtemp.at(data().F(i, j)).insert(i);

					}


				}
				shape->VF = OVtemp;
				shapesList->push_back(shape);
				return true;
			}

			IGL_INLINE bool Viewer::save_mesh_to_file(
				const std::string& mesh_file_name_string)
			{
				// first try to load it with a plugin
				//for (unsigned int i = 0; i<plugins.size(); ++i)
				//  if (plugins[i]->save(mesh_file_name_string))
				//    return true;

				size_t last_dot = mesh_file_name_string.rfind('.');
				if (last_dot == std::string::npos)
				{
					// No file type determined
					std::cerr << "Error: No file extension found in " <<
						mesh_file_name_string << std::endl;
					return false;
				}
				std::string extension = mesh_file_name_string.substr(last_dot + 1);
				if (extension == "off" || extension == "OFF")
				{
					return igl::writeOFF(
						mesh_file_name_string, data().V, data().F);
				}
				else if (extension == "obj" || extension == "OBJ")
				{
					Eigen::MatrixXd corner_normals;
					Eigen::MatrixXi fNormIndices;

					Eigen::MatrixXd UV_V;
					Eigen::MatrixXi UV_F;

					return igl::writeOBJ(mesh_file_name_string,
						data().V,
						data().F,
						corner_normals, fNormIndices, UV_V, UV_F);
				}
				else
				{
					// unrecognized file type
					printf("Error: %s is not a recognized file type.\n", extension.c_str());
					return false;
				}
				return true;
			}

			IGL_INLINE bool Viewer::load_scene()
			{
				std::string fname = igl::file_dialog_open();
				if (fname.length() == 0)
					return false;
				return load_scene(fname);
			}

			IGL_INLINE bool Viewer::load_scene(std::string fname)
			{
				// igl::deserialize(core(),"Core",fname.c_str());
				igl::deserialize(data(), "Data", fname.c_str());
				return true;
			}

			IGL_INLINE bool Viewer::save_scene()
			{
				std::string fname = igl::file_dialog_save();
				if (fname.length() == 0)
					return false;
				return save_scene(fname);
			}

			IGL_INLINE bool Viewer::save_scene(std::string fname)
			{
				//igl::serialize(core(),"Core",fname.c_str(),true);
				igl::serialize(data(), "Data", fname.c_str());

				return true;
			}

			IGL_INLINE void Viewer::open_dialog_load_mesh()
			{
				std::string fname = igl::file_dialog_open();

				if (fname.length() == 0)
					return;

				this->load_mesh_from_file(fname.c_str());
			}

			IGL_INLINE void Viewer::open_dialog_save_mesh()
			{
				std::string fname = igl::file_dialog_save();

				if (fname.length() == 0)
					return;

				this->save_mesh_to_file(fname.c_str());
			}

			IGL_INLINE ViewerData& Viewer::data(int mesh_id /*= -1*/)
			{
				assert(!data_list.empty() && "data_list should never be empty");
				int index;
				if (mesh_id == -1)
					index = selected_data_index;
				else
					index = mesh_index(mesh_id);

				assert((index >= 0 && index < data_list.size()) &&
					"selected_data_index or mesh_id should be in bounds");
				return data_list[index];
			}

			IGL_INLINE const ViewerData& Viewer::data(int mesh_id /*= -1*/) const
			{
				assert(!data_list.empty() && "data_list should never be empty");
				int index;
				if (mesh_id == -1)
					index = selected_data_index;
				else
					index = mesh_index(mesh_id);

				assert((index >= 0 && index < data_list.size()) &&
					"selected_data_index or mesh_id should be in bounds");
				return data_list[index];
			}

			IGL_INLINE int Viewer::append_mesh(bool visible /*= true*/)
			{
				assert(data_list.size() >= 1);

				data_list.emplace_back();
				selected_data_index = data_list.size() - 1;
				data_list.back().id = next_data_id++;
				//if (visible)
				//    for (int i = 0; i < core_list.size(); i++)
				//        data_list.back().set_visible(true, core_list[i].id);
				//else
				//    data_list.back().is_visible = 0;
				return data_list.back().id;
			}

			IGL_INLINE bool Viewer::erase_mesh(const size_t index)
			{
				assert((index >= 0 && index < data_list.size()) && "index should be in bounds");
				assert(data_list.size() >= 1);
				if (data_list.size() == 1)
				{
					// Cannot remove last mesh
					return false;
				}
				data_list[index].meshgl.free();
				data_list.erase(data_list.begin() + index);
				if (selected_data_index >= index && selected_data_index > 0)
				{
					selected_data_index--;
				}

				return true;
			}

			IGL_INLINE size_t Viewer::mesh_index(const int id) const {
				for (size_t i = 0; i < data_list.size(); ++i)
				{
					if (data_list[i].id == id)
						return i;
				}
				return 0;
			}

			Eigen::Matrix4d Viewer::CalcParentsTrans(int indx)
			{
				Eigen::Matrix4d prevTrans = Eigen::Matrix4d::Identity();

				for (int i = indx; parents[i] >= 0; i = parents[i])
				{
					//std::cout << "parent matrix:\n" << scn->data_list[scn->parents[i]].MakeTrans() << std::endl;
					prevTrans = data_list[parents[i]].MakeTransd() * prevTrans;
				}

				return prevTrans;
			}
			
			
			void Viewer::reset()
			{
				/*
				our:this function will make all the neccesery calculations
					that have been written in the given article, and with puts
					this information in the data structures that belongs to the
					chosen object
			
				
				
				*/
				MatrixXi F = shapesList->at(selected_data_index)->OF; //Our: F holds the faces, when removing triangles F is going to change and FO is going to be constant and never change
				MatrixXd V = shapesList->at(selected_data_index)->OV;//Our: V holds the verteces , when removing triangles V is going to change and VO is going to be constant and never change
				shapesList->at(selected_data_index)->V = V;
				shapesList->at(selected_data_index)->F = F;
				vector<set<int>> VF = shapesList->at(selected_data_index)->VF;
				MatrixXd& C = shapesList->at(selected_data_index)->C;
				MatrixXi& E = shapesList->at(selected_data_index)->E;
				VectorXi& EMAP = shapesList->at(selected_data_index)->EMAP;
				VectorXi& EQ = shapesList->at(selected_data_index)->EQ;
				igl::min_heap< std::tuple<double, int, int>>& QHeap = shapesList->at(selected_data_index)->QHeap;
				vector<MatrixXd>& Q_of_V = shapesList->at(selected_data_index)->Q_of_V;
				MatrixXi& EI = shapesList->at(selected_data_index)->EI;
				MatrixXi& EF = shapesList->at(selected_data_index)->EF;
				QHeap = {};
				/*
				this function will initialize all of the data structures of a chosen object
				(EMAP,EV,EI and so on...)
				*/
				edge_flaps(F, E, EMAP, EF, EI);
				C.resize(E.rows(), V.cols());
				EQ = Eigen::VectorXi::Zero(E.rows());
				int vOneIndex, vTwoIndex;
				double cost;
				const size_t ROW = 4;
				const size_t COL = 4;
				VectorXd normalVec;
				Vector4d vectorCord;
				Vector3d vectorLocation;
				MatrixXd Q_Value, Q_ValueTemp;
				Q_Value.resize(ROW, COL);

				/*
				we clculate the Q matrix for each vertex and push it to 
				the Q_of_V for each index in its index.
				Q_of_V is a vector that holds matrixes, while vector in index i
				holds the Q of vertex i
				*/
				for (int i = 0; i < V.rows(); i++)
				{
					Q_Value.setZero();
					set<int> setOfFaces = VF.at(i);
					for (set<int>::iterator k = setOfFaces.begin(); k != setOfFaces.end(); k++) {

						int  f = *k;
						double  a, b, c, d;

						normalVec = data().F_normals.row(f).normalized();

						//our:  print normal x,y,z
						a = normalVec[0];
						b = normalVec[1];
						c = normalVec[2];
						d = a * -V(i, 0) + b * -V(i, 1) + c * -V(i, 2);
						//row 0
						Q_Value(0, 0) = Q_Value(0, 0) + a * a;
						Q_Value(0, 1) = Q_Value(0, 1) + a * b;
						Q_Value(0, 2) = Q_Value(0, 2) + a * c;
						Q_Value(0, 3) = Q_Value(0, 3) + a * d;

						//row 1 
						Q_Value(1, 0) = Q_Value(1, 0) + a * b;
						Q_Value(1, 1) = Q_Value(1, 1) + b * b;
						Q_Value(1, 2) = Q_Value(1, 2) + b * c;
						Q_Value(1, 3) = Q_Value(1, 3) + b * d;


						//row 2

						Q_Value(2, 0) = Q_Value(2, 0) + a * c;
						Q_Value(2, 1) = Q_Value(2, 1) + b * c;
						Q_Value(2, 2) = Q_Value(2, 2) + c * c;
						Q_Value(2, 3) = Q_Value(2, 3) + c * d;

						//row 3 

						Q_Value(3, 0) = Q_Value(3, 0) + a * d;
						Q_Value(3, 1) = Q_Value(3, 1) + b * d;
						Q_Value(3, 2) = Q_Value(3, 2) + c * d;
						Q_Value(3, 3) = Q_Value(3, 3) + d * d;



					}

					Q_of_V.push_back(Q_Value);
					// cout << ",\n";
					 /*
					 for (size_t i = 0; i < Q_Value.rows(); i++)
					 {
						 for (size_t j = 0; j < Q_Value.row(i).cols(); j++)
						 {
							 cout << ","<< Q_Value(i,j);
						 }
						 cout << ",\n";
					 }
					 cout << ",\n";
					 */
				}
				for (size_t i = 0; i < E.rows(); i++)
				{
					vOneIndex = E(i, 0);
					vTwoIndex = E(i, 1);
					Q_Value = Q_of_V.at(vOneIndex) + Q_of_V.at(vTwoIndex);
					Q_ValueTemp = Q_Value;
					vectorCord = findOptVValue(Q_ValueTemp, vOneIndex, vTwoIndex);
					cost = (vectorCord.transpose() * Q_Value * vectorCord)(0, 0);
					vectorLocation(0) = vectorCord(0);
					vectorLocation(1) = vectorCord(1);
					vectorLocation(2) = vectorCord(2);
					QHeap.emplace(cost, i, 0);
					C.row(i) = vectorLocation;

				}

			
				data().clear();
				data().set_mesh(V, F);
				data().set_face_based(true);
				data().dirty = 157;

			}

			//our:find the new location of the new vertex
			VectorXd Viewer::findOptVValue(MatrixXd Q_Value, int vOneIndex, int vTwoIndex)
			{
				
				Vector4d ZeroVector(0, 0, 0, 1);
				Q_Value.row(3) = ZeroVector;


				if (Q_Value.determinant() != 0)
				{
					return (Q_Value.inverse() * ZeroVector);
				}
				return 0.5 * (V.row(vOneIndex) + V.row(vTwoIndex));

			}
			void Viewer::removeEdge()
			{
				MatrixXd &V = shapesList->at(selected_data_index)->V;
				 MatrixXi &F = shapesList->at(selected_data_index)->F;
				MatrixXd &C = shapesList->at(selected_data_index)->C;;
				MatrixXi & E = shapesList->at(selected_data_index)->E;
				VectorXi & EMAP = (shapesList->at(selected_data_index)->EMAP);
				VectorXi & EQ = shapesList->at(selected_data_index)->EQ;
				igl::min_heap< std::tuple<double, int, int>>& QHeap = shapesList->at(selected_data_index)->QHeap;
				vector<MatrixXd> & Q_of_V = shapesList->at(selected_data_index)->Q_of_V;
				MatrixXi& EI = shapesList->at(selected_data_index)->EI;
				MatrixXi & EF = shapesList->at(selected_data_index)->EF;
				if (!QHeap.empty())
				{
					bool something_collapsed = false;
					// collapse edge
					const int max_iter = std::ceil(0.01 * QHeap.size());
					for (int j = 0; j < max_iter; j++)
					{
						/*
						our:this function removes  the edge with the minimal cost,
						we will get the minimal cost from the QHeap and we will look on
						the 2 vereces from both sides of the selected edge, we will start
						mapping the edges that touches each one of the verteces of this selected
						edge, the edges we mapped created tringles that share the minimal cost edge
						(as we can see in the video that simulating this process)
						those tringles will be removed in the circulation
						
						
						*/
						if (!collapse_edge(V, F, E, EMAP, EF, EI, QHeap, EQ, C,Q_of_V,true))
						{
							break;
						}
						something_collapsed = true;
						num_collapsed++;
					}

					if (something_collapsed)
					{
						num_collapsed = 0;
						data().clear();
						data().set_mesh(V, F);
						data().set_face_based(true);
						data().dirty = 157;
					}
				}


			}
		} // end namespace
	} // end namespace
}
