
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "observer.hpp"
#include "pressiodemoapps/advection_diffusion2d.hpp"
#include <chrono>
#include "yaml-cpp/parser.h"
#include "yaml-cpp/yaml.h"
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;

template<class T = void>
pressio::ode::StepScheme string_to_ode_scheme(const std::string & schemeString)
{

  if (schemeString == "ForwardEuler"){
    return pressio::ode::StepScheme::ForwardEuler;
  }
  else if (schemeString == "RungeKutta4"){
    return pressio::ode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "RK4"){
    return pressio::ode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "SSPRungeKutta3"){
    return pressio::ode::StepScheme::SSPRungeKutta3;
  }
  else if (schemeString == "BDF1"){
    return pressio::ode::StepScheme::BDF1;
  }
  else if (schemeString == "CrankNicolson"){
    return pressio::ode::StepScheme::CrankNicolson;
  }
  else if (schemeString == "BDF2"){
    return pressio::ode::StepScheme::BDF2;
  }
  else{
    throw std::runtime_error("string_to_ode_scheme: Invalid odeScheme");
  }

}

pda::InviscidFluxReconstruction
stringToInviscidRecScheme(const std::string & string)
{
  if (string == "FirstOrder"){
    return pda::InviscidFluxReconstruction::FirstOrder;
  }
  else if (string == "Weno3"){
    return pda::InviscidFluxReconstruction::Weno3;
  }
  else if (string == "Weno5"){
    return pda::InviscidFluxReconstruction::Weno5;
  }

  return {};
}

template<class T>
void write_vector_to_ascii_file(std::string fileName, const T & vec)
{
  std::ofstream file; file.open(fileName);
  for (size_t i=0; i<vec.size(); i++){
    file << std::setprecision(15) << vec(i) << " \n";
  }
  file.close();
}

std::vector<int> read_int_vector_from_ascii(const std::string & fileName)
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
  return v;
}

template<class T>
void fill_view_from_ascii_file(std::string fileName, const T & M)
{
  std::vector<std::vector<double>> A0;
  pressio::utils::read_ascii_matrix_stdvecvec(fileName, A0, M.extent(1));

  for (int i=0; i<M.extent(0); ++i){
    for (int j=0; j<M.extent(1); ++j){
      M(i,j) = A0[i][j];
    }
  }
}

template<class FomObjType>
struct OdRomGappyConstK
{
  using rom_state_t = Eigen::VectorXd;

  // required public
  using scalar_type   = double;
  using state_type    = rom_state_t;
  using velocity_type = rom_state_t;

  const FomObjType & fomObj_;
  int K_ = 0;
  int numTiles_ = 0;
  int totalNumModes_ = 0;

  Kokkos::View<int> count_sm_;
  Kokkos::View<int> count_stm_;
  Kokkos::View<double***> phis_  = {};
  Kokkos::View<double***> projs_ = {};
  Eigen::VectorXd fomState_;
  mutable Eigen::VectorXd fomVeloc_;
  // Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> > fomStateView_;

  OdRomGappyConstK(const FomObjType & fomObj, int K, YAML::Node & node)
    : fomObj_(fomObj), K_(K),
      fomState_(fomObj.totalDofStencilMesh()),
      fomVeloc_(fomObj.totalDofSampleMesh()),
      numTiles_(node["numTiles"].as<int>()),
      count_sm_("c1", 16),
      count_stm_("c2", 16),
      phis_("phis", numTiles_, 1, K),
      projs_("proj", numTiles_, 1, K)
  {
    totalNumModes_ = K_ * numTiles_;

    // const auto meshDir = node["meshDir"].as<std::string>();
    // const auto count_sam = read_int_vector_from_ascii(meshDir+"/count_sm.txt");
    // const auto count_stm = read_int_vector_from_ascii(meshDir+"/count_stm.txt");
    // for (int k=0; k<16; ++k){
    //   count_sm_(k) = count_sam_[k];
    //   count_stm_(k) = count_stm_[k];
    //   std::cout << count_sm_(k) << " " << count_stm_(k) << std::endl;
    // }

    // const std::string phiOnStencilMeshDir  = node["phiOnStencilDir"].as<std::string>();
    // Kokkos::resize(phis_, numTiles_, 650, K);
    // for (int i=0; i<numTiles_; ++i){
    //   const int r0 = 0; const int r1 = count_stm_[i];
    //   auto sv = Kokkos::subview(phis_, i, std::make_pair(r0,r1), Kokkos::ALL());
    //   const std::string currFile = phiOnStencilMeshDir + "/phi_on_stencil_p_" + std::to_string(i) + ".txt";
    //   fill_view_from_ascii_file(currFile, sv);
    // }

    // const std::string projsDir = node["projectorDir"].as<std::string>();
    // Kokkos::resize(projs_, numTiles_, 312, K);
    // for (int i=0; i<numTiles_; ++i){
    //   auto sv = Kokkos::subview(projs_, i, Kokkos::ALL(), Kokkos::ALL());
    //   const std::string currFile = projsDir + "/projector_p_" + std::to_string(i) + ".txt";
    //   fill_view_from_ascii_file(currFile, sv);
    // }

    // const std::string projectorDirPath = node["projectorDir"];
    // const std::string refStateFile = node["refStateFile"];
  }

  rom_state_t createRomState() const {
    rom_state_t r(totalNumModes_);
    r.setZero();
    return r;
  }

  velocity_type createVelocity() const {
    velocity_type v(totalNumModes_);
    v.setZero();
    return v;
  }

  void velocity(const state_type & romState,
		scalar_type evaltime,
		velocity_type & romRhs)  const
  {
    // do reconstruction

    // using v_t = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    // v_t x_view (romState.data(), romState.size());

    // Kokkos::parallel_for(numTiles,
			 // KOKKOS_LAMBDA(const std::size_t i)
			 // {
			 //   const int r0 = 0; const int r1 = count_stm_(i);
			 //   auto myPhi = Kokkos::subview(phis_, i, std::make_pair(r0,r1), Kokkos::ALL());

			 //   const int romStateSpanBegin = i*K_;
			 //   const int romStateSpanEnd   = romStateSpanBegin + K_;
			 //   auto myRomState = Kokkos::subview(x_view, std::make_pair(romStateSpanBegin, romStateSpanEnd));

			 //   // const int fomStateSpanBegin = i*stencilDofsCountPerTile;
			 //   // const int fomStateSpanEnd   = fomStateSpanBegin + stencilDofsCountPerTile;
			 //   // auto myFomState = Kokkos::subview(fomState, std::make_pair(fomStateSpanBegin, fomStateSpanEnd));

			 //   // using notr = KokkosBatched::Trans::NoTranspose;
			 //   // using alg  = KokkosBatched::Algo::Gemv::Blocked;
			 //   // KokkosBatched::SerialGemv<notr, alg>::invoke(1.0, myPhi, myRomState, 0.0, myFomState);
			 // });

    // fomObj_.velocity(fomState_, evaltime, fomVeloc_);
    // eval fom veloc
    // projection
  }
};


int main(int argc, char *argv[])
{
  const std::string inputFile = argv[1];
  auto node = YAML::LoadFile(inputFile);

  Kokkos::initialize();
  {
    const auto meshDir = node["meshDir"].as<std::string>() + "/pda_sm";
    const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(meshDir);

    const auto inviscidFluxRecStr = node["inviscidFluxReconstruction"].as<std::string>();
    const auto inviscidSchemeE = stringToInviscidRecScheme(inviscidFluxRecStr);
    const auto viscSchemeE = pda::ViscousFluxReconstruction::FirstOrder;
    const auto pulseSpread = node["pulsespread"].as<double>();
    const auto pulseMag    = node["pulsemag"].as<double>();
    const auto diffusion   = node["diffusion"].as<double>();
    auto appObj = pda::create_burgers_2d_problem_eigen(meshObj, inviscidSchemeE, viscSchemeE,
						       pulseMag, pulseSpread, diffusion,
						       -0.15, -0.3);
    using app_t = decltype(appObj);

    const auto modesPerTile = read_int_vector_from_ascii("./modes_per_tile.txt");
    const auto K = modesPerTile[0];
    std::cout << "K = " << K << '\n';
    if (!std::all_of(modesPerTile.cbegin()+1, modesPerTile.cend(), [&](const auto val){ return val == K; })){
      throw std::runtime_error("not unique K");
    }

    OdRomGappyConstK<app_t> odromObj(appObj, K, node);

    const auto startTime = static_cast<double>(0);
    const auto finalTime = node["finalTime"].as<double>();
    const auto dt = node["dt"].as<double>();
    const auto numSteps = static_cast<int>(finalTime/dt);

    auto romState = odromObj.createRomState();
    write_vector_to_ascii_file("initial_state.txt", romState);

    const auto odeSchemeE = string_to_ode_scheme( node["odeScheme"].as<std::string>() );
    auto stepperObj = pode::create_explicit_stepper(odeSchemeE, romState, odromObj);
    auto t1 = std::chrono::high_resolution_clock::now();
    pode::advance_n_steps(stepperObj, romState, startTime, dt, numSteps);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > fs = t2 - t1;
    std::cout << "elapsed " << fs.count() << '\n';
    std::cout << "one-step " << fs.count()/(double) numSteps << std::endl;

    //write_vector_to_ascii_file("final_state.txt", romState);
  }
  Kokkos::finalize();

  return 0;
}
