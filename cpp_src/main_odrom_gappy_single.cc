
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"

namespace pressio{ namespace ops{

void update(Kokkos::View<double**> & mv, const double &a,
	    const Kokkos::View<double**> & mv1, const double &b)
{
  Kokkos::parallel_for(mv.extent(0),
		       KOKKOS_LAMBDA(const std::size_t i)
		       {
			 for (int j=0; j<mv.extent(1); ++j){
			   mv(i,j) = a*mv(i,j) + b*mv1(i,j);
			 }
		       });
}

void update(Kokkos::View<double**> & mv, const double &a,
	    const Kokkos::View<double**> & mv1, const double &b,
	    const Kokkos::View<double**> & mv2, const double &c)
{
  Kokkos::parallel_for(mv.extent(0),
		       KOKKOS_LAMBDA(const std::size_t i)
		       {
			 for (int j=0; j<mv.extent(1); ++j){
			   mv(i,j) = a*mv(i,j) + b*mv1(i,j) + c*mv2(i,j);
			 }
		       });
}

void update(Kokkos::View<double**> & mv, const double &a,
	    const Kokkos::View<double**> & mv1, const double &b,
	    const Kokkos::View<double**> & mv2, const double &c,
	    const Kokkos::View<double**> & mv3, const double &d)
{
  Kokkos::parallel_for(mv.extent(0),
		       KOKKOS_LAMBDA(const std::size_t i)
		       {
			 for (int j=0; j<mv.extent(1); ++j){
			   mv(i,j) = a*mv(i,j) + b*mv1(i,j) + c*mv2(i,j) + d*mv3(i,j);
			 }
		       });
}

void update(Kokkos::View<double**> & mv, const double &a,
	    const Kokkos::View<double**> & mv1, const double &b,
	    const Kokkos::View<double**> & mv2, const double &c,
	    const Kokkos::View<double**> & mv3, const double &d,
	    const Kokkos::View<double**> & mv4, const double &e)
{
  Kokkos::parallel_for(mv.extent(0),
		       KOKKOS_LAMBDA(const std::size_t i)
		       {
			 for (int j=0; j<mv.extent(1); ++j){
			   mv(i,j) = a*mv(i,j) + b*mv1(i,j) + c*mv2(i,j) + d*mv3(i,j) + e*mv4(i,j);
			 }
		       });
}

}}

#include "pressio/type_traits.hpp"
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "observer.hpp"
#include "pressiodemoapps/advection_diffusion2d.hpp"
#include "pressiodemoapps/swe2d.hpp"
#include <chrono>
#include "yaml-cpp/parser.h"
#include "yaml-cpp/yaml.h"

#include "odrom_single.hpp"
#include "odrom_multi_gemm.hpp"
#include "odrom_multi_gemv.hpp"

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

pressiodemoapps::InviscidFluxReconstruction
stringToInviscidRecScheme(const std::string & string)
{
  if (string == "FirstOrder"){
    return pressiodemoapps::InviscidFluxReconstruction::FirstOrder;
  }
  else if (string == "Weno3"){
    return pressiodemoapps::InviscidFluxReconstruction::Weno3;
  }
  else if (string == "Weno5"){
    return pressiodemoapps::InviscidFluxReconstruction::Weno5;
  }

  return {};
}

int main(int argc, char *argv[])
{
  namespace pode = pressio::ode;
  namespace pda = pressiodemoapps;

  const std::string inputFile = argv[1];
  int nth = std::atoi(argv[2]);
  int loops = std::atoi(argv[3]);
  std::cout << nth << " " << loops << '\n';
  auto node = YAML::LoadFile(inputFile);

  Kokkos::initialize();
  {
    const auto meshDir = node["meshDir"].as<std::string>() + "/pda_sm";
    const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(meshDir);

    const auto inviscidFluxRecStr = node["inviscidFluxReconstruction"].as<std::string>();

    //const auto inviscidSchemeE    = stringToInviscidRecScheme(inviscidFluxRecStr);
    //const auto viscSchemeE        = pda::ViscousFluxReconstruction::FirstOrder;
    //const auto pulseSpread      = node["pulsespread"].as<double>();
    // const auto pulseMag           = node["pulsemag"].as<double>();
    // const auto diffusion          = node["diffusion"].as<double>();
    // using fom_t = decltype(pda::create_burgers_2d_problem_eigen(meshObj,
    // 								inviscidSchemeE,
    // 								viscSchemeE,
    // 								0., 0., 0., 0., 0.));
    // auto appObj = pda::create_burgers_2d_problem_eigen(meshObj, inviscidSchemeE, viscSchemeE,
    // 						       pulseMag, 0.65, diffusion, -0.15, -0.3);
    //appObj.setNumThreads(nth);

    const auto inviscidSchemeE = stringToInviscidRecScheme("Weno5");
    auto appObj = pda::create_problem_eigen(meshObj, pda::Swe2d::SlipWall, inviscidSchemeE);
    using fom_t = decltype(appObj);

    OdRomSingle<fom_t> odromObj(appObj, node);

    const auto startTime  = static_cast<double>(0);
    const auto finalTime  = node["finalTime"].as<double>();
    const auto timeStepSz = node["dt"].as<double>();
    const auto numSteps   = static_cast<int>(finalTime/timeStepSz);

    auto romState = odromObj.createRomState();

    {
      auto veloc = odromObj.createVelocity();
      // warm up
      odromObj.velocity(romState, 0., veloc);

      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<loops; ++i){
	odromObj.velocity(romState, 0., veloc);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > fs = t2 - t1;
      std::cout << "elapsed " << fs.count() << '\n';
      std::cout << "one-velo " << fs.count()/(double) loops << std::endl;
    }

    #if 1
          const auto odeSchemeE = string_to_ode_scheme( node["odeScheme"].as<std::string>() );
          auto stepperObj = pode::create_explicit_stepper(odeSchemeE, romState, odromObj);
          auto t1 = std::chrono::high_resolution_clock::now();
          pode::advance_n_steps(stepperObj, romState, startTime, timeStepSz, numSteps);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::chrono::duration< double > fs = t2 - t1;
          std::cout << "elapsed " << fs.count() << '\n';
          std::cout << "one-step " << fs.count()/(double) numSteps << std::endl;
          write_rank1_view_to_ascii_file("rom_state.txt", romState);
    #endif

  }
  Kokkos::finalize();

  return 0;
}
