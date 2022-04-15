#ifndef ODROM_MULTI_GEMV_NESTED_HPP
#define ODROM_MULTI_GEMV_NESTED_HPP

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBatched_Gemm_Serial_Impl.hpp"

#include "io.hpp"

template<class FomType>
struct OdRomMultiGemvNested
{

  // required
  using scalar_type   = double;
  using state_type    = Kokkos::View<double**>;
  using velocity_type = state_type;

  const std::vector<FomType> & fomObjs_;
  const int numFoms_ = {};
  const int numTiles_ = 0;
  const int totTilesAllRuns_ = 0;

  Kokkos::View<int*> modesPerTile_;
  int totalModes_ = {};
  int maxK_ = {};
  int minK_ = {};

  int numChunks_ = {};
  int maxChunkSize_ = {};
  Kokkos::View<double***> phis_;
  Kokkos::View<int*> phiNumRowsPerChunk_;
  int maxRowsPhiOnStencil_ = {};

  Kokkos::View<double***> projs_;
  Kokkos::View<int*> projNumRowsPerTile_;
  int maxRowsProjectors_ = {};

  mutable Eigen::MatrixXd fomStatesForPdaEvaluation_;
  Kokkos::LayoutStride fomStateLay_;
  Kokkos::View<double**, Kokkos::LayoutStride> fomStateView_;

  mutable Eigen::MatrixXd fomVelocitiesForPdaEvaluation_;
  Kokkos::LayoutStride fomVelocLay_;
  Kokkos::View<double**, Kokkos::LayoutStride> fomVelocView_;

  Kokkos::View<int*> romStateSpanStarts_;
  Kokkos::View<int*> fomStateSpanStarts_;
  Kokkos::View<int*> fomVelocSpanStarts_;

  Kokkos::View<double**> refStates_;

  OdRomMultiGemvNested(const std::vector<FomType> & fomObjs,
		 const YAML::Node & node)
    : fomObjs_(fomObjs),
      numFoms_(fomObjs_.size()),
      numTiles_(node["numTiles"].as<int>()),
      totTilesAllRuns_(numTiles_ * numFoms_),
      //
      modesPerTile_("modesPerTile", numTiles_),
      phis_("phi", numTiles_, 1, 1),
      phiNumRowsPerChunk_("phiNumRowsPerChunk", numTiles_),
      projs_("projs", numTiles_, 1, 1),
      projNumRowsPerTile_("projNumRowsPerChunk_", 1),
      //
      fomStatesForPdaEvaluation_(fomObjs[0].totalDofStencilMesh(), numFoms_),
      fomStateLay_(fomStatesForPdaEvaluation_.rows(), 1,
		   numFoms_, fomStatesForPdaEvaluation_.rows()),
      fomStateView_(fomStatesForPdaEvaluation_.data(), fomStateLay_),
      //
      fomVelocitiesForPdaEvaluation_(fomObjs[0].totalDofSampleMesh(), numFoms_),
      fomVelocLay_(fomVelocitiesForPdaEvaluation_.rows(), 1,
		   numFoms_, fomVelocitiesForPdaEvaluation_.rows()),
      fomVelocView_(fomVelocitiesForPdaEvaluation_.data(), fomVelocLay_),
      //
      romStateSpanStarts_("romStateSpanStarts", numTiles_),
      fomStateSpanStarts_("fomStateSpanStarts", 1),
      fomVelocSpanStarts_("fomVelocSpanStarts", numTiles_),
      refStates_("refStates", fomObjs[0].totalDofStencilMesh(), numFoms_)
  {
    fomStatesForPdaEvaluation_.setZero();
    fomVelocitiesForPdaEvaluation_.setZero();

    for (int k=0; k<numFoms_; ++k){
      const auto fomIc = fomObjs[k].initialCondition();
      for (int i=0; i<fomIc.size(); ++i){
	refStates_(i,k) = fomIc(i);
      }
    }

    read_integers_from_ascii_into_view("./modes_per_tile.txt", modesPerTile_);
    // for (int i=0; i<numTiles_; ++i){
    //   std::cout << "tileId = " << i << " modes = " << modesPerTile_(i) << '\n';
    // }
    // std::cout << "\n";

    // compute total num of modes
    Kokkos::parallel_reduce("sumModes", numTiles_,
			    KOKKOS_LAMBDA (const int i, int& update) {
			      update += modesPerTile_(i);
			    }, totalModes_);

    // find what is the min/max K over all tiles
    Kokkos::parallel_reduce("maxK", numTiles_,
			     KOKKOS_LAMBDA (const int& i, int& val) {
			       const int K = modesPerTile_(i);
			       if(K > val) val = K;
			     }, Kokkos::Max<int>(maxK_));
    Kokkos::parallel_reduce("minK", numTiles_,
			     KOKKOS_LAMBDA (const int& i, int& val) {
			       const int K = modesPerTile_(i);
			       if(K < val) val = K;
			     }, Kokkos::Min<int>(minK_));

    // read phi on stencil mesh within each PHYSICAL tile
    const auto phiOnStencilMeshDir = node["phiOnStencilDir"].as<std::string>();
    Kokkos::View<int*> phiNumRowsPerTile("phiNumRowsPerTile", numTiles_);
    read_integers_from_ascii_into_view(phiOnStencilMeshDir + "/rows_per_tile.txt", phiNumRowsPerTile);
    maxRowsPhiOnStencil_ = read_single_int_from_ascii(phiOnStencilMeshDir+"/max_num_rows.txt");

    Kokkos::View<double***> phiTmp_("phiTmp", numTiles_, maxRowsPhiOnStencil_, maxK_);
    for (int i=0; i<numTiles_; ++i)
    {
      const auto myK = modesPerTile_(i);
      auto sv = Kokkos::subview(phiTmp_, i, Kokkos::ALL(), std::make_pair(0, myK));
      const auto currFile = phiOnStencilMeshDir + "/phi_on_stencil_p_" + std::to_string(i) + ".txt";
      read_phi_on_stencil_from_ascii_file(currFile, sv);
    }

    numChunks_ = 60;
    Kokkos::resize(phiNumRowsPerChunk_, numChunks_*numTiles_);
    int count=0;
    for (int i=0; i<numTiles_; ++i)
    {
      const auto myPhiOnStencilNumRows = phiNumRowsPerTile(i);
      const auto nominalChunkSize = int(myPhiOnStencilNumRows/numChunks_);
      //std::cout << "tile = " << i << " " << myPhiOnStencilNumRows << '\n';
      if  (myPhiOnStencilNumRows < nominalChunkSize){
	throw std::runtime_error("cannot have maxRowsPhiOnStencil_ <= chunkSize_");
      }

      int sum=0;
      for (int k=0; k<numChunks_-1; ++k){
	phiNumRowsPerChunk_(count++) = nominalChunkSize;
	//	std::cout << "count = " << count-1 << " " << nominalChunkSize << '\n';
	sum += nominalChunkSize;
      }
      phiNumRowsPerChunk_(count++) = myPhiOnStencilNumRows-sum;
      //std::cout << "count = " << count-1 << " " << myPhiOnStencilNumRows-sum << '\n';
    }

    // find max chunk size
    Kokkos::parallel_reduce("maxChSize", phiNumRowsPerChunk_.extent(0),
			     KOKKOS_LAMBDA (const int& i, int& val) {
			       const int v = phiNumRowsPerChunk_(i);
			       if(v > val) val = v;
			     }, Kokkos::Max<int>(maxChunkSize_));
    //std::cout << "maxChunkSize = " << maxChunkSize_ << '\n';

    // now we fill the phi for each chunk for each tile
    Kokkos::resize(phis_, numTiles_*numChunks_, maxChunkSize_, maxK_);
    count = 0;
    for (int i=0; i<numTiles_; ++i)
    {
      auto phiTile = Kokkos::subview(phiTmp_, i, Kokkos::ALL(), Kokkos::ALL());

      int startCount = 0;
      for (int k=0; k<numChunks_; ++k){
	const int currChunkNumRowsPhi = phiNumRowsPerChunk_(count);
	const int end = startCount + currChunkNumRowsPhi;

	// std::cout << "phiChunk for tile = " << i
	// 	  << " chunk_k = " << k
	// 	  << " chunk_numRowsPhi = " << currChunkNumRowsPhi
	// 	  << " (s,e) " << startCount << " " << end
	// 	  << '\n';

	auto myPhiChunk = Kokkos::subview(phis_, count++, Kokkos::ALL(), Kokkos::ALL());
	for (int row=0; row<currChunkNumRowsPhi; ++row){
	  for (int col=0; col<phiTile.extent(1); ++col){
	    myPhiChunk(row,col) = phiTile(row+startCount, col);
	  }
	}
	startCount += currChunkNumRowsPhi;
      }
    }

    Kokkos::resize(fomStateSpanStarts_, numTiles_*numChunks_);
    Kokkos::parallel_scan(fomStateSpanStarts_.extent(0),
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = phiNumRowsPerChunk_(i);
			    if (final) { fomStateSpanStarts_(i) = update; }
			    update += val_i;
			  });
    // for (int i=0; i<fomStateSpanStarts_.extent(0); ++i){
    //   std::cout << "fomStateSS(i) = " << i << " " << fomStateSpanStarts_(i) << '\n';
    // }

    // read projectors for each tile
    const auto projectorsDir = node["projectorDir"].as<std::string>();
    read_integers_from_ascii_into_view(projectorsDir + "/rows_per_tile.txt", projNumRowsPerTile_);
    maxRowsProjectors_ = read_single_int_from_ascii(projectorsDir+"/max_num_rows.txt");
    Kokkos::resize(projs_, numTiles_, maxRowsProjectors_, maxK_);
    for (int i=0; i<numTiles_; ++i)
    {
      const auto myK = modesPerTile_(i);
      auto sv = Kokkos::subview(projs_, i, Kokkos::ALL(), std::make_pair(0, myK));
      const auto currFile = projectorsDir + "/projector_p_" + std::to_string(i) + ".txt";
      read_projector_from_ascii_file(currFile, sv);
    }
    //std::cout << "\n";

    Kokkos::parallel_scan(numTiles_,
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = projNumRowsPerTile_(i);
			    if (final) { fomVelocSpanStarts_(i) = update; }
			    update += val_i;
			  });

    Kokkos::parallel_scan(numTiles_,
			  KOKKOS_LAMBDA (const int i, int & update, const bool final){
			    const int val_i = modesPerTile_(i);
			    if (final) { romStateSpanStarts_(i) = update; }
			    update += val_i;
			  });

    // for (int i=0; i<numTiles_; ++i){
    //   std::cout << "romStateSS(i) = " << i << " " << romStateSpanStarts_(i)
    // 		<< " "
    // 		<< "fomStateSS(i) = " << i << " " << fomStateSpanStarts_(i)
    // 		<< " "
    // 		<< "fomVelocSS(i) = " << i << " " << fomVelocSpanStarts_(i)
    // 		<< '\n';
    // }

    printInfo();
  }

  void printInfo()
  {
    std::cout << "--------------------------------------------------\n";
    std::cout << "totRealizations             = " << numFoms_ << '\n';
    std::cout << "totNumTilesAllRealizations_ = " << totTilesAllRuns_ << '\n';
    std::cout << "totModesPerRealization      = " << totalModes_ << '\n';
    std::cout << "{min,max}NumModesOverTiles  = " << minK_ << " " << maxK_ << '\n';
    std::cout << "maxRowsPhiOnStencil         = " << maxRowsPhiOnStencil_ << '\n';
    std::cout << "maxRowsProjectors           = " << maxRowsProjectors_ << '\n';
    std::cout << "\n";

    std::cout << "romStateSizeAllRealiz = " << totalModes_*numFoms_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "romStateSizePerRealiz = " << totalModes_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";

    std::cout << "romRhsSizeAllRealiz   = " << totalModes_*numFoms_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "romRhsSizePerRealiz   = " << totalModes_*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";

    std::cout << "fomStateSizeAllRealiz = " <<
      fomStatesForPdaEvaluation_.rows()*fomStatesForPdaEvaluation_.cols()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "fomStateSizePerRealiz = " <<
      fomStatesForPdaEvaluation_.rows()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "fomRhsSizeAllRealiz   = " <<
      fomVelocitiesForPdaEvaluation_.rows()*fomStatesForPdaEvaluation_.cols()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";
    std::cout << "fomRhsSizePerRealiz   = " <<
      fomVelocitiesForPdaEvaluation_.rows()*sizeof(scalar_type)/(double) 1e6 << " (MB)\n";

    std::cout << "--------------------------------------------------";
    std::cout << "\n";
  }

  state_type createRomState() const {
    state_type res("crs", totalModes_, numFoms_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  velocity_type createVelocity() const {
    state_type res("cv", totalModes_, numFoms_);
    KokkosBlas::fill(res, 0.);
    return res;
  }

  void velocity(const state_type & romState,
		scalar_type evaltime,
		velocity_type & romRhs) const
  {
    doReconstructionTeam(romState);
    KokkosBlas::axpy(1., refStates_, fomStateView_);

//     Kokkos::parallel_for(numFoms_,
// 			   KOKKOS_LAMBDA(const int i)
// 			   {
// 			     auto fomState = fomStatesForPdaEvaluation_.col(i);
// 			     auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(i);
// 			     fomObjs_[i].velocity(fomState, evaltime, fomVeloc);
// 			   });
// #pragma omp barrier

    auto fomState = fomStatesForPdaEvaluation_.col(0);
    auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(0);
    fomObjs_[0].velocity(fomState, evaltime, fomVeloc);

// #pragma omp parallel num_threads(numFoms_)
//     {
// #pragma omp for
//       for (int i=0; i<numFoms_; ++i){
// 	auto fomState = fomStatesForPdaEvaluation_.col(i);
// 	auto fomVeloc = fomVelocitiesForPdaEvaluation_.col(i);
// 	fomObjs_[i].velocity(fomState, evaltime, fomVeloc);
//       }
//     }

// #pragma omp barrier
    doProjectionTeam(romRhs);
    //#pragma omp barrier
  }

private:

  void doReconstructionTeam(const state_type & romState) const
  {
    const auto N = numTiles_ * numChunks_ * numFoms_;

    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(N, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type member)
			 {
			   const int index   = member.league_rank();
			   const int chunkId = index % (numTiles_ * numChunks_);
			   const int tileId  = int(chunkId / numChunks_);
			   const int realization = index/(numTiles_ * numChunks_);

			   const int myK = modesPerTile_[tileId];

			   const int myPhi_s0 = phiNumRowsPerChunk_(chunkId);
			   auto myPhi = Kokkos::subview(phis_, chunkId,
							std::make_pair(0, myPhi_s0),
							std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomState = Kokkos::subview(romState,
							     std::make_pair(rs_b, rs_e),
							     realization);

			   const int fs_b = fomStateSpanStarts_(chunkId);
			   const int fs_e = fs_b + myPhi_s0;
			   auto myFomState = Kokkos::subview(fomStateView_,
							     std::make_pair(fs_b, fs_e),
							     realization);

			   assert(myRomState.extent(0) == myPhi.extent(1));
			   assert(myFomState.extent(0) == myPhi.extent(0));

			   using notr = KokkosBatched::Trans::NoTranspose;
			   using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			   KokkosBatched::TeamVectorGemv<
			     member_type, notr, alg>::invoke(member, 1.0, myPhi, myRomState,
							     0.0, myFomState);
			 });
  }

  void doProjectionTeam(velocity_type & romRhs) const
  {
    typedef Kokkos::DefaultExecutionSpace SpT;
    Kokkos::TeamPolicy<SpT> policy(totTilesAllRuns_, Kokkos::AUTO());
    typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

    Kokkos::parallel_for(policy,
			 KOKKOS_LAMBDA(const member_type member)
			 {
			   const int realization = member.league_rank()/numTiles_;
			   const int tileId = member.league_rank() % numTiles_;

			   const int myK = modesPerTile_[tileId];

			   const int myProj_s0 = projNumRowsPerTile_(tileId);
			   auto myProj = Kokkos::subview(projs_, tileId,
							 std::make_pair(0, myProj_s0),
							 std::make_pair(0, myK));

			   const int rs_b = romStateSpanStarts_(tileId);
			   const int rs_e = rs_b  + myK;
			   auto myRomRhs = Kokkos::subview(romRhs,
							   std::make_pair(rs_b, rs_e),
							   realization);

			   const int fv_b = fomVelocSpanStarts_(tileId);
			   const int fv_e = fv_b + myProj_s0;
			   auto myFomVeloc = Kokkos::subview(fomVelocView_,
							     std::make_pair(fv_b, fv_e),
							     realization);

			   assert(myRomState.extent(0) == myProj.extent(1));
			   assert(myFomVeloc.extent(0) == myProj.extent(0));

			   using tr = KokkosBatched::Trans::Transpose;
			   using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			   KokkosBatched::TeamVectorGemv<
			     member_type, tr, alg>::invoke(member, 1.0, myProj,
							   myFomVeloc, 0.0, myRomRhs);
			 });
  }

};

#endif
