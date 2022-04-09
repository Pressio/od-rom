
#include <utility>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include "Kokkos_Random.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Serial_Impl.hpp"
#include "CLI11.hpp"


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

template<class T1, class T2>
void naive_range(int K, int numTiles,
		 int phiRowsPerTile,
		 int loopCount,
		 T1 & phis,
		 T2 & romState)
{
  Kokkos::View<double*>   fomState("fomY", phiRowsPerTile*numTiles);
  KokkosBlas::fill(fomState, double(0.0));

  std::cout << "naive_range\n";
  std::cout << " numTiles     = " << numTiles << '\n';
  std::cout << " romStateSize = " << romState.extent(0) << '\n';
  std::cout << " fomStateSize = " << fomState.extent(0) << '\n';

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int count=0; count<loopCount; ++count)
    {
      Kokkos::parallel_for(numTiles,
			   KOKKOS_LAMBDA(const int i)
			   {
			     auto myPhi = Kokkos::subview(phis, i,
							  Kokkos::ALL(),
							  Kokkos::ALL());

			     const int romStateSpanBegin = i*K;
			     const int romStateSpanEnd   = romStateSpanBegin + K;
			     const auto r1 = std::make_pair(romStateSpanBegin, romStateSpanEnd);
			     auto myRomState = Kokkos::subview(romState, r1);

			     const int fomStateSpanBegin = i*phiRowsPerTile;
			     const int fomStateSpanEnd   = fomStateSpanBegin + phiRowsPerTile;
			     const auto r2 = std::make_pair(fomStateSpanBegin, fomStateSpanEnd);
			     auto myFomState = Kokkos::subview(fomState, r2);

			     using notr = KokkosBatched::Trans::NoTranspose;
			     using alg  = KokkosBatched::Algo::Gemv::Blocked;
			     KokkosBatched::SerialGemv<
			       notr, alg>::invoke(1.0, myPhi,
						  myRomState, 0.0, myFomState);
			   });
      Kokkos::fence();
    }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t2 - t1;
  std::cout << " one-eval " << fs.count()/(double) loopCount << std::endl;

  write_rank1_view_to_ascii_file("naive_range.txt", fomState);
}

template<class T1, class T2>
void naive_team(int K, int numTiles,
		 int phiRowsPerTile,
		 int loopCount,
		 T1 & phis,
		 T2 & romState)
{
  Kokkos::View<double*>   fomState("fomY", phiRowsPerTile*numTiles);
  KokkosBlas::fill(fomState, double(0.0));

  std::cout << "naive_team\n";
  std::cout << " numTiles     = " << numTiles << '\n';
  std::cout << " romStateSize = " << romState.extent(0) << '\n';
  std::cout << " fomStateSize = " << fomState.extent(0) << '\n';

  typedef Kokkos::DefaultExecutionSpace SpT;
  Kokkos::TeamPolicy<SpT> policy(numTiles, Kokkos::AUTO());
  typedef Kokkos::TeamPolicy<SpT>::member_type member_type;

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int count=0; count<loopCount; ++count)
    {
      Kokkos::parallel_for(policy,
			   KOKKOS_LAMBDA(const member_type & member)
			   {
			     const std::size_t i = member.league_rank();
			     auto myPhi = Kokkos::subview(phis, i,
							  Kokkos::ALL(),
							  Kokkos::ALL());

			     const int romStateSpanBegin = i*K;
			     const int romStateSpanEnd   = romStateSpanBegin + K;
			     const auto r1 = std::make_pair(romStateSpanBegin, romStateSpanEnd);
			     auto myRomState = Kokkos::subview(romState, r1);

			     const int fomStateSpanBegin = i*phiRowsPerTile;
			     const int fomStateSpanEnd   = fomStateSpanBegin + phiRowsPerTile;
			     const auto r2 = std::make_pair(fomStateSpanBegin, fomStateSpanEnd);
			     auto myFomState = Kokkos::subview(fomState, r2);

			     using notr = KokkosBatched::Trans::NoTranspose;
			     // must be nonblocked here because blocked is not impl yet
			     using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			     KokkosBatched::TeamVectorGemv<
			       member_type, notr, alg>::invoke(member, 1.0, myPhi, myRomState,
							     0.0, myFomState);
			   });
      Kokkos::fence();
    }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t2 - t1;
  std::cout << " one-eval " << fs.count()/(double) loopCount << std::endl;

  write_rank1_view_to_ascii_file("naive_team.txt", fomState);
}


template<class T1, class T2>
void nested(int K, int numTiles,
	    int phiRowsPerTile,
	    int loopCount,
	    int chunkSize_p,
	    int chunkSize_q,
	    T1 & phis,
	    T2 & romState)
{

  Kokkos::View<double*> fomState("fomY", phiRowsPerTile*numTiles);
  KokkosBlas::fill(fomState, double(0.0));

  std::cout << "nested\n";
  std::cout << " numTiles     = " << numTiles << '\n';
  std::cout << " romStateSize = " << romState.extent(0) << '\n';
  std::cout << " fomStateSize = " << fomState.extent(0) << '\n';

  if (K % chunkSize_p != 0){
    throw std::runtime_error("dfdfd");
  }
  if (phiRowsPerTile % chunkSize_q != 0){
    throw std::runtime_error("aaaadfd");
  }

  const auto p = K / chunkSize_p;
  const auto q = phiRowsPerTile / chunkSize_q;
  std::cout << "p = " << p << " q = " << q << '\n';
  std::cout << "q*numTiles = " << q*numTiles << '\n';

  const auto blocksPerTile = p*q;
  std::cout << "blocksPerTile = " << blocksPerTile << '\n';
  const auto N = blocksPerTile * numTiles;
  std::cout << "N = " << N << '\n';

  // for (int bId=0; bId<blocksPerTile*2; ++bId){
  //   const int tileId = bId / blocksPerTile;
  //   const int i = bId % q;
  //   const int j = (bId - blocksPerTile*tileId) / q;
  //   std::cout << " bId = " << bId << " "
  // 	      << " tileId = " << tileId
  // 	      << " i = " << i
  // 	      << " j = " << j
  //             << " rs,e = " << i*chunkSize_q << " " << i*chunkSize_q+chunkSize_q
  // 	      << " cs,e = " << j*chunkSize_p << " " << j*chunkSize_p+chunkSize_p
  // 	      << std::endl;
  // }

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int count=0; count<loopCount; ++count)
    {

#if 1
      for (int j=0; j<p; ++j){
	Kokkos::parallel_for(q*numTiles, KOKKOS_LAMBDA(const std::size_t bId){
	    const int tileId = bId / q;
	    const int i = bId % q;

	    const auto rs = i*chunkSize_q;
	    const auto cs = j*chunkSize_p;
	    auto myPhi = Kokkos::subview(phis, tileId,
					 std::make_pair(rs, rs+chunkSize_q),
					 std::make_pair(cs, cs+chunkSize_p));

	    const int romStateSpanBegin = tileId*K + j*chunkSize_p;
	    const int romStateSpanEnd   = romStateSpanBegin + chunkSize_p;
	    const auto r1 = std::make_pair(romStateSpanBegin, romStateSpanEnd);
	    auto myRomState = Kokkos::subview(romState, r1);

	    const int fomStateSpanBegin = tileId*phiRowsPerTile + i*chunkSize_q;
	    const int fomStateSpanEnd   = fomStateSpanBegin + chunkSize_q;
	    const auto r2 = std::make_pair(fomStateSpanBegin, fomStateSpanEnd);
	    auto myFomState = Kokkos::subview(fomState, r2);

	    using notr = KokkosBatched::Trans::NoTranspose;
	    using alg  = KokkosBatched::Algo::Gemv::Blocked;
	    const auto beta = j==0 ? 0.0 : 1.0;
	    KokkosBatched::SerialGemv<notr, alg>::invoke(1.0, myPhi,
							 myRomState,
							 beta, myFomState);
	  });
	Kokkos::fence();
      }

#else

      Kokkos::View<double*> fomst2("fomst2", N*chunkSize_q);
      Kokkos::parallel_for(N,
			   KOKKOS_LAMBDA(const std::size_t bId)
			   {
			     const int tileId = bId / blocksPerTile;
			     const int i = bId % q;
			     const int j = (bId - blocksPerTile*tileId) / q;

			     const auto rs = i*chunkSize_q;
			     const auto cs = j*chunkSize_p;
			     auto myPhi = Kokkos::subview(phis, tileId,
							  std::make_pair(rs, rs+chunkSize_q),
							  std::make_pair(cs, cs+chunkSize_p));

			     const int romStateSpanBegin = tileId*K + j*chunkSize_p;
			     const int romStateSpanEnd   = romStateSpanBegin + chunkSize_p;
			     const auto r1 = std::make_pair(romStateSpanBegin, romStateSpanEnd);
			     auto myRomState = Kokkos::subview(romState, r1);

			     const int fomStateSpanBegin = bId*chunkSize_q;
			     const int fomStateSpanEnd   = fomStateSpanBegin + chunkSize_q;
			     const auto r2 = std::make_pair(fomStateSpanBegin, fomStateSpanEnd);
			     auto myFomState = Kokkos::subview(fomst2, r2);

			     using notr = KokkosBatched::Trans::NoTranspose;
			     using alg  = KokkosBatched::Algo::Gemv::Unblocked;
			     const auto beta = j==0 ? 0.0 : 1.0;
			     KokkosBatched::SerialGemv<notr, alg>::invoke(1.0, myPhi,
									  myRomState,
									  beta, myFomState);
			   });
      Kokkos::fence();

#endif

    }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t2 - t1;
  std::cout << " one-eval " << fs.count()/(double) loopCount << std::endl;
  write_rank1_view_to_ascii_file("nested.txt", fomState);
}

int main(int argc, char *argv[])
{
  CLI::App app;
  int K = 0;
  int numTiles = 0;
  int phiRowsPerTile = 0;
  int loopCount = 10;
  int chunkSize_p = 10;
  int chunkSize_q = 10;

  app.add_option("-n", numTiles);
  app.add_option("-K", K);
  app.add_option("-M", phiRowsPerTile);
  app.add_option("-l", loopCount);
  app.add_option("-p", chunkSize_p);
  app.add_option("-q", chunkSize_q);
  CLI11_PARSE(app, argc, argv);

  Kokkos::initialize();
  {
    Kokkos::View<double***> phis("phis", numTiles, phiRowsPerTile, K);
    Kokkos::View<double*>   romState("romState", numTiles*K);

    Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace> random(13718);
    Kokkos::fill_random(phis, random, double(0.5));

    //KokkosBlas::fill(romState, double(0.0));
    //auto sv = Kokkos::subview(romState, std::make_pair(K, 2*K));
    Kokkos::fill_random(romState, random, double(0.5));
    //write_rank1_view_to_ascii_file("rom_st.txt", romState);

    // for (int i=0; i<numTiles; ++i){
    //   auto tile_A = Kokkos::subview(phis, i, Kokkos::ALL(), Kokkos::ALL());
    //   auto tile_x = Kokkos::subview(romState, std::make_pair(i*K, i*K+K));
    //   Kokkos::View<double*> y2("y2", tile_A.extent(0));
    //   KokkosBlas::gemv("N", 1., tile_A, tile_x, 0.0, y2);
    //   write_rank1_view_to_ascii_file("gemv" + std::to_string(i) +".txt", y2);
    //   Kokkos::fence();
    // }

    naive_range(K, numTiles, phiRowsPerTile, loopCount, phis, romState);
    naive_team(K, numTiles, phiRowsPerTile, loopCount, phis, romState);
    nested(K, numTiles, phiRowsPerTile, loopCount,
     	   chunkSize_p, chunkSize_q, phis, romState);
  }

  Kokkos::finalize();
  return 0;
}
