
#include "pressio/ode_steppers_explicit.hpp"
#include "pressio/ode_steppers_implicit.hpp"
#include "pressio/ode_advancers.hpp"
#include "observer.hpp"
#include "pressiodemoapps/advection_diffusion2d.hpp"
#include <chrono>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "CLI11.hpp"

namespace pda = pressiodemoapps;
namespace pode = pressio::ode;

template<class T = void>
pode::StepScheme string_to_ode_scheme(const std::string & schemeString)
{

  if (schemeString == "ForwardEuler"){
    return pode::StepScheme::ForwardEuler;
  }
  else if (schemeString == "RungeKutta4"){
    return pode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "RK4"){
    return pode::StepScheme::RungeKutta4;
  }
  else if (schemeString == "SSPRungeKutta3"){
    return pode::StepScheme::SSPRungeKutta3;
  }
  else if (schemeString == "BDF1"){
    return pode::StepScheme::BDF1;
  }
  else if (schemeString == "CrankNicolson"){
    return pode::StepScheme::CrankNicolson;
  }
  else if (schemeString == "BDF2"){
    return pode::StepScheme::BDF2;
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
  for (size_t i=0; i<vec.extent(0); i++){
    file << std::setprecision(15) << vec(i) << " \n";
  }
  file.close();
}

template<class FomObjType>
struct OdRomProxy
{

  // required
  using scalar_type   = double;
  using state_type    = Eigen::VectorXd;
  using velocity_type = Eigen::VectorXd;

  const FomObjType & fomObj_;
  int K_ = 0;
  int numTiles_ = 0;
  mutable state_type    fomStateForPdaEvaluation_;
  mutable velocity_type fomVelocForPdaEvaluation_;

  const int sampleMeshCountPerTile_;
  const int stencilMeshCountPerTile_;
  Kokkos::View<double***> projectors_;
  Kokkos::View<double***> phis_;
  Kokkos::View<double*>   romState_;
  mutable state_type    fomStateFakeManaging_;
  mutable velocity_type fomVelocFakeManaging_;

  using unma_t = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
  Kokkos::View<double*, Kokkos::HostSpace, unma_t> fomStateView_;
  Kokkos::View<double*, Kokkos::HostSpace, unma_t> fomVelocView_;

  OdRomProxy(const FomObjType & fomObj,  int K, int numTiles,
	     int sampleMeshCountPerTile, int stencilMeshCountPerTile)
    : fomObj_(fomObj),
      K_(K),
      numTiles_(numTiles),
      //
      fomStateForPdaEvaluation_(fomObj.initialCondition()),
      fomVelocForPdaEvaluation_(fomObj.createVelocity()),
      sampleMeshCountPerTile_(sampleMeshCountPerTile),
      stencilMeshCountPerTile_(stencilMeshCountPerTile),
      //
      projectors_("projector", numTiles, 1, 1),
      phis_("phi", numTiles, 1, 1),
      romState_("romState", numTiles*K),
      //
      fomVelocFakeManaging_(sampleMeshCountPerTile*numTiles),
      fomStateFakeManaging_(stencilMeshCountPerTile*numTiles),
      fomStateView_(fomStateFakeManaging_.data(), fomStateFakeManaging_.size()),
      fomVelocView_(fomVelocFakeManaging_.data(), fomVelocFakeManaging_.size())
  {
    Kokkos::resize(projectors_, numTiles, sampleMeshCountPerTile,  K);
    Kokkos::resize(phis_,       numTiles, stencilMeshCountPerTile, K);
    fomStateFakeManaging_.setZero();
    fomVelocFakeManaging_.setZero();

    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> random(13718);
    Kokkos::fill_random(phis_, random, double(1.0));
    Kokkos::fill_random(projectors_, random, double(1.0));
    Kokkos::fill_random(romState_, random, double(1.0));
    Kokkos::fill_random(fomStateView_, random, double(1.0));
    Kokkos::fill_random(fomVelocView_, random, double(1.0));
  }

  state_type createRomState() const {
    state_type r(K_*numTiles_);
    r.setZero();
    return r;
  }

  velocity_type createVelocity() const {
    velocity_type v(K_*numTiles_);
    v.setZero();
    return v;
  }

  void velocity(const state_type & romState,
		scalar_type evaltime,
		velocity_type & romRhs) const
  {
    doReconstruction();
    Kokkos::fence();
    fomObj_.velocity(fomStateForPdaEvaluation_, 0., fomVelocForPdaEvaluation_);
    doProjection();
    Kokkos::fence();
  }

private:
  void doReconstruction() const
  {

    // fomState = phi * romState for each tile
    Kokkos::parallel_for(numTiles_,
			 KOKKOS_LAMBDA(const std::size_t i)
			 {
			   auto myPhi = Kokkos::subview(phis_,
							i, Kokkos::ALL(), Kokkos::ALL());

			   const int romStateSpanBegin = i*K_;
			   const int romStateSpanEnd   = romStateSpanBegin + K_;
			   auto myRomState = Kokkos::subview(romState_,
							     std::make_pair(romStateSpanBegin,
									    romStateSpanEnd));

			   const int fomStateSpanBegin = i*stencilMeshCountPerTile_;
			   const int fomStateSpanEnd   = fomStateSpanBegin + stencilMeshCountPerTile_;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fomStateSpanBegin,
									    fomStateSpanEnd));

			   assert(myRomState.extent(0) == K_);
			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemv::Blocked;
			   KokkosBatched::SerialGemv<notr, alg>::invoke(1.0, myPhi, myRomState,
									0.0, myFomState);
			 });

    // auto fomStateTile0 = Kokkos::subview(fomStateView_, std::make_pair(0, stencilMeshCountPerTile_));
    // write_vector_to_ascii_file("fomStateTile0_batch.txt", fomStateTile0);
    // auto phi0 = Kokkos::subview(phis_, 0, Kokkos::ALL(), Kokkos::ALL());
    // auto romStateTile0 = Kokkos::subview(romState_, std::make_pair(0, K_));
    // KokkosBlas::gemv("N", 1., phi0, romStateTile0, 0.0, fomStateTile0);
    // write_vector_to_ascii_file("fomStateTile0_normal.txt", fomStateTile0);
  }

  void doProjection() const
  {
    Kokkos::parallel_for(numTiles_,
			 KOKKOS_LAMBDA(const std::size_t i)
			 {
			   auto myProj = Kokkos::subview(projectors_,
							 i, Kokkos::ALL(), Kokkos::ALL());

			   const int romStateSpanBegin = i*K_;
			   const int romStateSpanEnd   = romStateSpanBegin + K_;
			   auto myRomState = Kokkos::subview(romState_,
							     std::make_pair(romStateSpanBegin,
									    romStateSpanEnd));

			   const int fomVelocSpanBegin = i*sampleMeshCountPerTile_;
			   const int fomVelocSpanEnd   = fomVelocSpanBegin + sampleMeshCountPerTile_;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fomVelocSpanBegin,
									    fomVelocSpanEnd));

			   assert(myRomState.extent(0) == K_);
			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using notr = KokkosBatched::Trans::Transpose;
			   using alg  = KokkosBatched::Algo::Gemv::Blocked;
			   KokkosBatched::SerialGemv<notr, alg>::invoke(1.0, myProj, myFomVeloc,
									0.0, myRomState);
			 });

    // auto romStateTile0 = Kokkos::subview(romState_, std::make_pair(0, K_));
    // write_vector_to_ascii_file("romStateTile0_batch.txt", romStateTile0);
    // auto fomVelocTile0 = Kokkos::subview(fomVelocView_, std::make_pair(0, sampleMeshCountPerTile_));
    // auto pro0 = Kokkos::subview(projectors_, 0, Kokkos::ALL(), Kokkos::ALL());
    // KokkosBlas::gemv("T", 1., pro0, fomVelocTile0, 0.0, romStateTile0);
    // write_vector_to_ascii_file("romStateTile0_normal.txt", romStateTile0);
  }
};

int main(int argc, char *argv[])
{
  CLI::App app;
  std::string meshDir = "void";
  int K = 0;
  int numTiles = 0;
  int loopCount = 100;
  int numSteps  = 100;
  app.add_option("-m", meshDir);
  app.add_option("-l", loopCount);
  app.add_option("-k", K);
  app.add_option("-p", numTiles);
  app.add_option("-n", numSteps);
  CLI11_PARSE(app, argc, argv);

  Kokkos::initialize();
  {

    const auto meshObj = pda::load_cellcentered_uniform_mesh_eigen(meshDir);
    const auto probE   = pda::AdvectionDiffusion2d::Burgers;
    const auto invSchE = stringToInviscidRecScheme("Weno5");
    const auto visSchE = pda::ViscousFluxReconstruction::FirstOrder;
    auto appObj        = pda::create_problem_eigen(meshObj, probE, invSchE, visSchE);
    const auto fomSampleMeshCount  = appObj.totalDofSampleMesh();
    const auto fomStencilMeshCount = appObj.totalDofStencilMesh();

    // approximate # of sample/stencil mesh cells per tile
    const auto sampleMeshCountPerTile  = (int) fomSampleMeshCount/numTiles;
    const auto stencilMeshCountPerTile = (int) fomStencilMeshCount/numTiles;
    std::cout << "sampleMeshCountPerTile  = " << sampleMeshCountPerTile << '\n';
    std::cout << "stencilMeshCountPerTile = " << stencilMeshCountPerTile << '\n';

    OdRomProxy<decltype(appObj)> proxy(appObj, K, numTiles,
				       sampleMeshCountPerTile,
				       stencilMeshCountPerTile);

    auto state = proxy.createRomState();

    // check velocity evaluation
    {
      auto veloc = proxy.createVelocity();
      // warmup
      proxy.velocity(state, 0.0, veloc);

      auto t1 = std::chrono::high_resolution_clock::now();
      for (int count=0; count<loopCount; ++count)
	{
	  proxy.velocity(state, 0.0, veloc);
	}
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > fs = t2 - t1;
      std::cout << "singleVeloEval " << fs.count()/(double) loopCount << std::endl;
    }

    // check real time stepping
    // this is not the same as just velo evaluation because it also involves
    // state updating so that has some effect
    {
      const auto odeSchemeE = string_to_ode_scheme("RK4");
      auto stepperObj = pode::create_explicit_stepper(odeSchemeE, state, proxy);
      auto t1 = std::chrono::high_resolution_clock::now();
      pode::advance_n_steps(stepperObj, state, 0., 0.0001, numSteps);
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration< double > fs = t2 - t1;
      std::cout << "singleTimeStep " << fs.count()/(double) numSteps << std::endl;
       }

  }
  Kokkos::finalize();
}
