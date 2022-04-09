
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "observer.hpp"
#include "pressiodemoapps/advection_diffusion2d.hpp"
#include <chrono>
#include "yaml-cpp/parser.h"
#include "yaml-cpp/yaml.h"

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

int main(int argc, char *argv[])
{
  const std::string inputFile = argv[1];
  auto node = YAML::LoadFile(inputFile);
  int nth   = std::atoi(argv[2]);
  int loops = std::atoi(argv[3]);
  std::cout << nth << " " << loops << '\n';

  const auto meshDir = node["meshDir"].as<std::string>();
  const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(meshDir);

  const auto inviscidFluxRecStr = node["inviscidFluxReconstruction"].as<std::string>();
  const auto inviscidSchemeE = stringToInviscidRecScheme(inviscidFluxRecStr);
  const auto viscSchemeE = pda::ViscousFluxReconstruction::FirstOrder;
  const auto pulseSpread = node["pulsespread"].as<double>();
  const auto pulseMag = node["pulsemag"].as<double>();
  const auto diffusion = node["diffusion"].as<double>();
  auto appObj = pda::create_burgers_2d_problem_eigen(meshObj, inviscidSchemeE, viscSchemeE,
						     pulseMag, pulseSpread, diffusion,
						     -0.15, -0.3);
  appObj.setNumThreads(nth);
  using app_t = decltype(appObj);

  const auto startTime = static_cast<double>(0);
  const auto finalTime = node["finalTime"].as<double>();
  const auto dt = node["dt"].as<double>();
  const auto numSteps = static_cast<int>(finalTime/dt);

  const int stateSamplingFreq = node["stateSamplingFreq"].as<int>();
  const int veloSamplingFreq = node["velocitySamplingFreq"].as<int>();
  StateObserver<typename app_t::state_type> stateObs("fom_snaps_state", stateSamplingFreq);
  VelocityObserver<> veloObs("fom_snaps_rhs", veloSamplingFreq);

  {
    auto state = appObj.initialCondition();
    auto veloc = appObj.createVelocity();
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<loops; ++i){
      appObj.velocity(state, 0., veloc);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration< double > fs = t2 - t1;
    std::cout << "elapsed " << fs.count() << '\n';
    std::cout << "one-velo " << fs.count()/(double) loops << std::endl;
  }

// #if 0
//   write_vector_to_ascii_file("initial_state.txt", state);
//   const auto odeSchemeE = string_to_ode_scheme( node["odeScheme"].as<std::string>() );
//   auto stepperObj = pode::create_explicit_stepper(odeSchemeE, state, appObj);
//   auto t1 = std::chrono::high_resolution_clock::now();
//   pode::advance_n_steps_and_observe(stepperObj, state, startTime,
// 				    dt, numSteps, stateObs, veloObs);
//   auto t2 = std::chrono::high_resolution_clock::now();
//   std::chrono::duration< double > fs = t2 - t1;
//   std::cout << "elapsed " << fs.count() << '\n';
//   std::cout << "one-step " << fs.count()/(double) numSteps << std::endl;
//   write_vector_to_ascii_file("final_state.txt", state);
// #endif

  return 0;
}
