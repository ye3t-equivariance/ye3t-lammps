struct YACEModelTestAccess {
 template<class T> static void vector_read(std::istream &in, std::vector<T>& v) {
   std::size_t n;in>>n; if(n>10000000)throw std::runtime_error("fixture count");v.resize(n);for(auto &x:v)in>>x;
 }
 static YACEModel load_fixture(const std::string &path) {
  std::ifstream in(path);std::string schema;in>>schema;if(schema!="ye3t_native_cpu_fixture_v1")throw std::runtime_error("fixture schema");
  YACEModel m;int ns;in>>ns>>m.spline_spacing_;if(ns<1||ns>10)throw std::runtime_error("fixture species");m.species_.resize(ns);m.bonds_.resize(ns*ns);
  for(auto &s:m.species_) {
   in>>s.element>>s.reference_energy>>s.embedding_scale>>s.density_safe_limit;
   int count;in>>count;s.channels.resize(count);
   for(auto &ch:s.channels){int kind;in>>kind>>ch.neighbor_species>>ch.radial>>ch.angular>>ch.magnetic;ch.kind=static_cast<YACEChannel::Kind>(kind);}
   auto &p=s.polynomial;in>>p.maximum_rank;
   vector_read(in,p.factor_offsets);vector_read(in,p.factor_indices);vector_read(in,p.factor_exponents);vector_read(in,p.monomial_coefficients);
   vector_read(in,p.descriptor_offsets);vector_read(in,p.descriptor_terms);vector_read(in,p.descriptor_coefficients);
   for(std::size_t i=1;i<p.factor_offsets.size();++i)p.maximum_term_factors=std::max(p.maximum_term_factors,p.factor_offsets[i]-p.factor_offsets[i-1]);
   build_monomial_dag(p);
   std::map<std::tuple<int,int,int,int,int>,int> sources;
   for(const auto &ch:s.channels){auto key=std::make_tuple(int(ch.kind),ch.neighbor_species,ch.radial,ch.angular,std::abs(ch.magnetic));auto ins=sources.emplace(key,int(sources.size()));if(ins.second){auto src=ch;src.magnetic=std::abs(src.magnetic);s.source_channels.push_back(src);}s.full_channel_sources.push_back(ins.first->second);s.full_channel_transforms.push_back(ch.magnetic<0?(std::abs(ch.magnetic)%2==0?1:-1):0);}
  }
  for(auto &b:m.bonds_) {
   in>>b.central_species>>b.neighbor_species>>b.radial_count>>b.angular_maximum>>b.radial_base_count>>b.radial_lambda>>b.cutoff>>b.cutoff_width;
   vector_read(in,b.radial_coefficients);build_splines(b,m.spline_spacing_);
   m.maximum_radial_base_count_=std::max(m.maximum_radial_base_count_,b.radial_base_count);m.maximum_contracted_width_=std::max(m.maximum_contracted_width_,b.contracted_width());m.maximum_angular_momentum_=std::max(m.maximum_angular_momentum_,b.angular_maximum);m.maximum_cutoff_=std::max(m.maximum_cutoff_,b.cutoff);
  }
  for(int i=0;i<ns;++i)for(std::size_t k=0;k<m.species_[i].source_channels.size();++k){const auto &c=m.species_[i].source_channels[k];auto &b=m.bonds_[i*ns+c.neighbor_species];if(c.kind==YACEChannel::RADIAL_BASE){b.radial_channel_outputs.push_back(k);b.radial_channel_indices.push_back(c.radial);}else{b.angular_channel_outputs.push_back(k);b.contracted_channel_indices.push_back(c.radial*(b.angular_maximum+1)+c.angular);b.angular_channel_nonnegative_indices.push_back(c.angular*(c.angular+1)/2+c.magnetic);}}
  if(!in)throw std::runtime_error("truncated fixture");return m;
 }
 static YACEModel two_species(YACEModel m) {
  if(m.species_count()!=1)throw std::runtime_error("twin fixture needs one species");
  auto second=m.species_[0];second.element="SyntheticSecond";
  second.reference_energy+=.5;second.embedding_scale*=.8;m.species_.push_back(second);
  const auto original=m.bonds_[0];m.bonds_.resize(4);
  for(int i=0;i<2;++i)for(int j=0;j<2;++j){auto b=original;b.central_species=i;b.neighbor_species=j;
   if(j==1){b.radial_channel_outputs.clear();b.radial_channel_indices.clear();b.angular_channel_outputs.clear();b.contracted_channel_indices.clear();b.angular_channel_nonnegative_indices.clear();}
   m.bonds_[i*2+j]=std::move(b);
  }
  return m;
 }
};
YACEModel make_two_species_native_fixture(YACEModel m) {return YACEModelTestAccess::two_species(std::move(m));}
YACEModel load_native_fixture(const std::string &path){return YACEModelTestAccess::load_fixture(path);}
// Expose the production compiler/residual builder to host GPU-admission tests.
YACESparsePolynomial compile_native_test_polynomial(YACESparsePolynomial p) {
 build_monomial_dag(p);return p;
}
YACESparsePolynomial native_test_without_descriptors(const YACESparsePolynomial &p,
                                                    const std::vector<int> &removed) {
 return without_descriptors(p,std::set<int>(removed.begin(),removed.end()));
}
} // namespace YE3T_LAMMPS
