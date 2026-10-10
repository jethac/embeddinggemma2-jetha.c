// Focused diagnostic for the observed I8MM CPU_REPACK audio failure.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>
static void require(bool v,const char *s) { if(!v) throw std::runtime_error(s); }
struct Q8 { ggml_fp16_t d; int8_t q[32]; }; static_assert(sizeof(Q8)==34,"Q8 layout");
static std::vector<float> run(ggml_backend_t backend,ggml_backend_buffer_type_t buft,
 const std::vector<Q8>& w,const std::vector<float>& a,int k,int n,int rows,int stride) {
 auto ctx=ggml_init({16*1024*1024,nullptr,true}); require(ctx,"context");
 auto wt=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,k,n);
 auto wb=ggml_backend_alloc_ctx_tensors_from_buft(ctx,buft); require(wb,"weight allocation");
 ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
 ggml_backend_tensor_set(wt,w.data(),0,w.size()*sizeof(Q8));
 if(std::strstr(ggml_backend_buft_name(buft),"REPACK")) require(wt->extra,"repack traits");
 auto storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,stride,rows);
 auto x=stride==k?storage:ggml_view_2d(ctx,storage,k,rows,stride*sizeof(float),0);
 auto out=ggml_mul_mat(ctx,wt,x);auto g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);
 auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buf,"graph allocation");
 ggml_backend_tensor_set(storage,a.data(),0,a.size()*sizeof(float));
 require(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
 std::vector<float> v(n*rows);ggml_backend_tensor_get(out,v.data(),0,v.size()*sizeof(float));
 std::printf("route buffer=%s rows=%d k=%d n=%d nb1=%zu extra=%d\n",ggml_backend_buft_name(buft),rows,k,n,x->nb[1],wt->extra!=nullptr);
 ggml_backend_buffer_free(buf);ggml_backend_buffer_free(wb);ggml_free(ctx);return v;
}
int main(int argc,char**argv) {try {
 require(argc==2,"runtime directory required");ggml_backend_load_all_from_path(argv[1]);
 auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);require(dev,"CPU device");
 auto reg=ggml_backend_dev_backend_reg(dev);auto backend=ggml_backend_dev_init(dev,nullptr);require(backend,"backend");
 auto threads=(ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads");require(threads,"threads");threads(backend,2);
 auto features=(ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_get_features");require(features,"features");bool i8mm=false;for(auto f=features(reg);f&&f->name;f++){std::printf("feature %s=%s\n",f->name,f->value);if(std::strcmp(f->name,"MATMUL_INT8")==0&&std::strcmp(f->value,"1")==0)i8mm=true;}require(i8mm,"actual I8MM selection");
 auto get=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts");require(get,"extra buffers");
 ggml_backend_buffer_type_t packed=nullptr;for(auto p=get(dev);p&&*p;++p)if(std::strstr(ggml_backend_buft_name(*p),"REPACK"))packed=*p;require(packed,"CPU_REPACK");
 const int k=512,n=768;std::vector<float> weights(k*n);for(size_t i=0;i<weights.size();i++)weights[i]=std::sin(i*.031f)+.37f*std::cos(i*.007f);
 std::vector<Q8>w(k*n/32);require(ggml_quantize_chunk(GGML_TYPE_Q8_0,weights.data(),w.data(),0,n,k,nullptr)==w.size()*sizeof(Q8),"weights quantization");
 auto quant=ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref;require(quant,"activation quantizer");bool good=true;
 for(int rows:{1,29})for(int pad:{0,16}) {
  int stride=k+pad;std::vector<float>a(stride*rows,-99);for(int r=0;r<rows;r++)for(int j=0;j<k;j++)a[r*stride+j]=std::sin((r*k+j)*.017f)+.21f*std::cos(j*.043f);
  std::vector<Q8> aq(k*rows/32);for(int r=0;r<rows;r++)quant(a.data()+r*stride,aq.data()+r*k/32,k);
  auto mapped=run(backend,ggml_backend_dev_buffer_type(dev),w,a,k,n,rows,stride);auto repack=run(backend,packed,w,a,k,n,rows,stride);
  for(int r=0;r<rows;r++) {double maxerr=0,maperr=0,dot=0,xnorm=0,ynorm=0;
   for(int c=0;c<n;c++) {float ref=0;for(int b=0;b<k/32;b++){auto &wb=w[c*k/32+b];auto &ab=aq[r*k/32+b];int sum=0;for(int j=0;j<32;j++)sum+=int(wb.q[j])*int(ab.q[j]);ref=std::fma(float(sum),ggml_fp16_to_fp32(wb.d)*ggml_fp16_to_fp32(ab.d),ref);}
    auto x=repack[r*n+c];auto y=mapped[r*n+c];maxerr=std::max(maxerr,double(std::abs(x-ref)));maperr=std::max(maperr,double(std::abs(y-ref)));dot+=double(x)*y;xnorm+=double(x)*x;ynorm+=double(y)*y;}
   double cos=dot/std::sqrt(xnorm*ynorm);std::printf("I8MM_REPACK rows=%d pad=%d row=%d packed_maxerr=%.9g mapped_maxerr=%.9g cosine=%.12g\n",rows,pad,r,maxerr,maperr,cos);if(!(cos>.999&&maxerr<.02))good=false;
  }
 }
 ggml_backend_free(backend);return good?0:2;
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
