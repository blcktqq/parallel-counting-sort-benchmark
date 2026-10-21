#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
using Keys=std::vector<int32_t>;
using Hist=std::vector<uint64_t>;
Hist histogram(const Keys& a,int lo,int k){
    Hist h(k,0);
    for(auto x:a){
        auto j=int64_t(x)-lo;
        if(j<0 || j>=k) throw std::runtime_error("Key out of range");
        ++h[size_t(j)];
    }
    return h;
}
Keys restore(const Hist& h,int lo){
    size_t n=0;for(auto c:h)n+=c;
    Keys out(n);size_t pos=0;
    for(size_t v=0;v<h.size();++v){
        std::fill_n(out.begin()+pos,h[v],int32_t(lo+int(v)));pos+=h[v];
    }
    return out;
}
Keys sequential(const Keys& a,int lo,int k){return restore(histogram(a,lo,k),lo);}

class Pool{
    std::vector<std::thread> threads;
    std::deque<std::function<void()>> jobs;
    std::mutex mutex;
    std::condition_variable work,ready;
    bool stop=false;size_t started=0;
public:
    explicit Pool(int p){
        for(int t=0;t<p;++t)threads.emplace_back([this]{
            {std::lock_guard<std::mutex> g(mutex);++started;ready.notify_one();}
            while(true){
                std::function<void()> job;
                {std::unique_lock<std::mutex> lock(mutex);
                 work.wait(lock,[this]{return stop || !jobs.empty();});
                 if(stop && jobs.empty())return;
                 job=std::move(jobs.front());jobs.pop_front();}
                job();
            }
        });
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock,[this,p]{return started==size_t(p);});
    }
    std::future<Hist> submit(Keys block,int lo,int k){
        auto task=std::make_shared<std::packaged_task<Hist()>>(
            [b=std::move(block),lo,k]{return histogram(b,lo,k);});
        auto result=task->get_future();
        {std::lock_guard<std::mutex> g(mutex);jobs.emplace_back([task]{(*task)();});}
        work.notify_one();return result;
    }
    ~Pool(){
        {std::lock_guard<std::mutex> g(mutex);stop=true;}
        work.notify_all();for(auto& thread:threads)thread.join();
    }
};
Keys parallel(const Keys& a,int lo,int k,int p,Pool& pool){
    // Every task receives a COPY, matching the algorithm used in Python.
    std::vector<Keys> blocks;
    for(int t=0;t<p;++t){
        auto first=a.size()*t/p,last=a.size()*(t+1)/p;
        blocks.emplace_back(a.begin()+first,a.begin()+last);
    }
    std::vector<std::future<Hist>> futures;
    for(auto& block:blocks)futures.push_back(pool.submit(std::move(block),lo,k));
    std::vector<Hist> partial;for(auto& f:futures)partial.push_back(f.get());
    Hist total(k,0);
    for(const auto& h:partial)for(int v=0;v<k;++v)total[v]+=h[v];
    return restore(total,lo);
}
Keys cold(const Keys& a,int lo,int k,int p){Pool pool(p);return parallel(a,lo,k,p,pool);}
void self_test(){
    Pool pool(3);int checks=0;
    for(int n:{0,1,2,7,100})for(int k:{1,7,256}){
        Keys a(n);for(int i=0;i<n;++i)a[i]=-k/2+(i*37+11)%k;
        for(int shape=0;shape<4;++shape){
            auto expected=a;std::sort(expected.begin(),expected.end());
            if(sequential(a,-k/2,k)!=expected || parallel(a,-k/2,k,3,pool)!=expected
               || cold(a,-k/2,k,3)!=expected)throw std::runtime_error("Self-test failed");
            ++checks;
            if(shape==0)std::sort(a.begin(),a.end());
            if(shape==1)std::reverse(a.begin(),a.end());
            if(shape==2)std::fill(a.begin(),a.end(),-k/2);
        }
    }
    std::cout<<"C++ PASS: "<<checks<<" datasets x 3 modes\n";
}
int main(int argc,char** argv){try{
    if(argc==2 && std::string(argv[1])=="--self-test"){self_test();return 0;}
    if(argc!=7)throw std::runtime_error("Usage: INPUT LO K REPEATS OUTPUT_CSV P_LIST");
    std::ifstream file(argv[1],std::ios::binary);if(!file)throw std::runtime_error("Input missing");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)),{});
    if(bytes.size()%4)throw std::runtime_error("Invalid i32 file");
    Keys a(bytes.size()/4);
    for(size_t i=0;i<a.size();++i){
        uint32_t u=uint32_t(bytes[4*i])|(uint32_t(bytes[4*i+1])<<8)|
                   (uint32_t(bytes[4*i+2])<<16)|(uint32_t(bytes[4*i+3])<<24);
        a[i]=int32_t(u<0x80000000u?int64_t(u):int64_t(u)-0x100000000LL);
    }
    int lo=std::stoi(argv[2]),k=std::stoi(argv[3]),repeats=std::stoi(argv[4]);
    std::vector<int> ps;std::stringstream ss(argv[6]);std::string part;
    while(std::getline(ss,part,','))ps.push_back(std::stoi(part));
    std::vector<std::unique_ptr<Pool>> pools;for(int p:ps)pools.push_back(std::make_unique<Pool>(p));
    auto expected=a;std::sort(expected.begin(),expected.end());
    struct Config{std::string mode;int index;};std::vector<Config> configs{{"seq",0}};
    for(size_t i=0;i<ps.size();++i){configs.push_back({"warm",int(i)});configs.push_back({"cold",int(i)});}
    auto run=[&](const Config& c){
        if(c.mode=="seq")return sequential(a,lo,k);
        if(c.mode=="warm")return parallel(a,lo,k,ps[c.index],*pools[c.index]);
        return cold(a,lo,k,ps[c.index]);
    };
    for(const auto& c:configs)if(run(c)!=expected)throw std::runtime_error("Warm-up mismatch");
    std::ofstream out(argv[5]);out<<"language,mode,p,repeat,ms\n"<<std::setprecision(12);
    std::mt19937 rng(42);
    for(int rep=1;rep<=repeats;++rep){
        std::shuffle(configs.begin(),configs.end(),rng);
        for(const auto& c:configs){
            auto start=std::chrono::steady_clock::now();auto result=run(c);
            auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(result!=expected)throw std::runtime_error("Result mismatch");
            out<<"C++,"<<c.mode<<','<<(c.mode=="seq"?1:ps[c.index])<<','<<rep<<','<<ms<<'\n';
        }
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
