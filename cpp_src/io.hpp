#ifndef IO_HPP
#define IO_HPP

#include <Kokkos_Core.hpp>
#include "pressio/utils.hpp"

template<class ViewType>
void write_rank1_view_to_ascii_file(std::string fileName,
				    const ViewType & vec)
{
  std::ofstream file; file.open(fileName);
  for (size_t i=0; i<vec.extent(0); i++){
    file << std::setprecision(15) << vec(i) << " \n";
  }
  file.close();
}

template<class VType>
void write_eigen_vector_to_ascii_file(std::string fileName,
				      const VType & vec)
{
  std::ofstream file; file.open(fileName);
  for (size_t i=0; i<vec.size(); i++){
    file << std::setprecision(15) << vec(i) << " \n";
  }
  file.close();
}

int read_single_int_from_ascii(const std::string & fileName){
  int result = {};
  std::ifstream source;
  source.open(fileName, std::ios_base::in);
  std::string line, colv;
  while (std::getline(source, line) ){
    std::istringstream in(line);
    in >> colv;
    result = std::atoi(colv.c_str());
  }
  source.close();
  return result;
}

void read_integers_from_ascii_into_view(const std::string & fileName,
					Kokkos::View<int*> dest)
{
  std::vector<int> v;
  std::ifstream source;
  source.open(fileName, std::ios_base::in);
  std::string line, colv;
  while (std::getline(source, line) ){
    std::istringstream in(line);
    in >> colv;
    v.push_back(std::atoi(colv.c_str()));
  }
  source.close();

  for (int i=0; i<v.size(); ++i){
    dest(i) = v[i];
  }
}

template<class ViewType>
void read_phi_on_stencil_from_ascii_file(std::string fileName,
					 ViewType & M)
{
  std::vector<std::vector<double>> A0;
  pressio::utils::read_ascii_matrix_stdvecvec(fileName, A0, M.extent(1));

  for (int i=0; i<A0.size(); ++i){
    for (int j=0; j<A0[0].size(); ++j){
      M(i,j) = A0[i][j];
    }
  }
}

template<class ViewType>
void read_projector_from_ascii_file(std::string fileName,
				    ViewType & M)
{
  std::vector<std::vector<double>> A0;
  pressio::utils::read_ascii_matrix_stdvecvec(fileName, A0, M.extent(1));

  for (int i=0; i<A0.size(); ++i){
    for (int j=0; j<A0[0].size(); ++j){
      M(i,j) = A0[i][j];
    }
  }
}

#endif
