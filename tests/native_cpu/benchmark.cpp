#include "ye3t_cpu_evaluator.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace YE3T_LAMMPS {
YACEModel load_native_fixture(const std::string &);
#ifndef LEGACY_ORIGINAL
struct YE3TCPUEvaluatorTestAccess {
 static void capacity(YE3TCPUEvaluator &e,int n) {e.source_tile_policy_.edge_capacity=n;
#ifndef PREVIOUS_EDGE_TILER
 e.reference_edge_tiling_=true;
#endif
}
};
#endif
}
using namespace YE3T_LAMMPS;
int main(int argc,char**argv) try {
 if(argc<5)throw std::runtime_error("usage: bench fixture centers center_batch auto|replay|whole [output] [--verify-only]");
 const std::string model_path=argv[1];const int n=std::stoi(argv[2]), chunk=std::stoi(argv[3]);if(n<1||chunk<1)throw std::runtime_error("counts");
 YACEModel model=load_native_fixture(model_path);YE3TCPUEvaluator evaluator(&model);
#ifdef LEGACY_ORIGINAL
 const int auto_capacity=0;
#else
 const int auto_capacity=evaluator.source_tile_edge_capacity();
 if(std::string(argv[4])=="whole")YE3TCPUEvaluatorTestAccess::capacity(evaluator,std::numeric_limits<int>::max());
 else if(std::string(argv[4])=="replay")YE3TCPUEvaluatorTestAccess::capacity(evaluator,auto_capacity);
 else if(std::string(argv[4])!="tile" && std::string(argv[4])!="auto")throw std::runtime_error("schedule must be auto, replay, or whole");
#endif
 // Repeated distorted BCC environments; this is a native E/VJP benchmark, NOT a LAMMPS/MPI run.
 const double lattice=3.3161146998079496; std::vector<std::array<double,3>> shell;
 const double cutoff=model.maximum_cutoff();int bound=int(std::ceil(cutoff/lattice))+1;
 for(int x=-bound;x<=bound;++x)for(int y=-bound;y<=bound;++y)for(int z=-bound;z<=bound;++z)for(int h=0;h<2;++h){std::array<double,3>r{{lattice*(x+.5*h),lattice*(y+.5*h),lattice*(z+.5*h)}};double rr=r[0]*r[0]+r[1]*r[1]+r[2]*r[2];if(rr>1e-10&&rr<(cutoff-.04)*(cutoff-.04))shell.push_back(r);}
 if(shell.empty())throw std::runtime_error("empty shell");int z=shell.size();
 std::vector<int> centers(n*z),types(n),neigh(n*z);std::vector<double>vectors(n*z*3),energies(n),grads(n*z*3);
 std::mt19937_64 rng(713);std::uniform_real_distribution<double>dis(-.01,.01);
 for(int i=0;i<n;++i){types[i]=i%model.species_count();for(int j=0;j<z;++j){int e=i*z+j;centers[e]=i%chunk;neigh[e]=(i+j)%model.species_count();for(int a=0;a<3;++a)vectors[e*3+a]=shell[j][a]+dis(rng);}}
 auto run=[&](){for(int b=0;b<n;b+=chunk){int m=std::min(chunk,n-b);evaluator.evaluate(m,types.data()+b,m*z,centers.data()+b*z,neigh.data()+b*z,vectors.data()+b*z*3,energies.data()+b,grads.data()+b*z*3);}};
 run();run(); std::vector<double>times; const int samples=3;
 const bool verify_only=argc>6 && std::string(argv[6])=="--verify-only";
 for(int s=0;s<samples;++s){if(verify_only){times.push_back(0);continue;}int it=0;auto start=std::chrono::steady_clock::now();double elapsed=0;do{run();++it;elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}while(elapsed<.12);times.push_back(elapsed/it);}
 std::sort(times.begin(),times.end());double et=std::accumulate(energies.begin(),energies.end(),0.),gn=0;for(double v:grads)gn+=v*v;
 std::cout<<std::setprecision(17)<<"{\"centers\":"<<n<<",\"neighbors\":"<<z<<",\"batch\":"<<chunk<<",\"schedule\":\""<<argv[4]<<"\",\"auto_edge_capacity\":"<<auto_capacity<<",\"lmax\":"<<model.maximum_angular_momentum()<<",\"channels\":"<<model.species(0).channels.size()<<",\"monomials\":"<<model.species(0).polynomial.monomial_coefficients.size()<<",\"powers\":"<<model.species(0).polynomial.power_channels.size()<<",\"binary_nodes\":"<<model.species(0).polynomial.binary_node_left.size()<<",\"binary_dag\":"<<model.species(0).polynomial.binary_dag<<",\"median_seconds\":"<<times[1]<<",\"min_seconds\":"<<times[0]<<",\"max_seconds\":"<<times[2]<<",\"scratch_bytes\":"<<evaluator.memory_usage()<<",\"energy_sum\":"<<et<<",\"gradient_norm_squared\":"<<gn<<"}\n";
 if(argc>5){std::ofstream out(argv[5]);out<<std::setprecision(17);for(double e:energies)out<<e<<'\n';for(double v:grads)out<<v<<'\n';}
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
