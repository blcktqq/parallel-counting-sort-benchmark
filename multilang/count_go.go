package main
import (
    "encoding/binary"
    "fmt"
    "math/rand"
    "os"
    "sort"
    "strconv"
    "strings"
    "sync"
    "time"
)
func histogram(a []int32,lo int32,k int) []uint64 {
    h:=make([]uint64,k)
    for _,x:=range a {
        j:=int64(x)-int64(lo)
        if j<0 || j>=int64(k) {panic("Key out of range")}
        h[j]++
    }
    return h
}
func restore(h []uint64,lo int32) []int32 {
    n:=0;for _,c:=range h {n+=int(c)}
    out:=make([]int32,n);pos:=0
    for v,c:=range h {
        stop:=pos+int(c)
        for pos<stop {out[pos]=lo+int32(v);pos++}
    }
    return out
}
func sequential(a []int32,lo int32,k int) []int32 {return restore(histogram(a,lo,k),lo)}
type Task struct {block []int32;lo int32;k int;reply chan []uint64}
type Pool struct {jobs chan Task;done sync.WaitGroup}
func newPool(p int) *Pool {
    pool:=&Pool{jobs:make(chan Task,p)}
    var ready sync.WaitGroup;ready.Add(p);pool.done.Add(p)
    for t:=0;t<p;t++ {go func(){
        defer pool.done.Done();ready.Done()
        for task:=range pool.jobs {task.reply<-histogram(task.block,task.lo,task.k)}
    }()}
    ready.Wait();return pool
}
func (pool *Pool) close(){close(pool.jobs);pool.done.Wait()}
func parallel(a []int32,lo int32,k,p int,pool *Pool) []int32 {
    blocks:=make([][]int32,p)
    for t:=0;t<p;t++ {
        first,last:=len(a)*t/p,len(a)*(t+1)/p
        blocks[t]=append([]int32{},a[first:last]...)
    }
    reply:=make(chan []uint64,p)
    for _,block:=range blocks {pool.jobs<-Task{block,lo,k,reply}}
    partial:=make([][]uint64,p)
    for t:=0;t<p;t++ {partial[t]=<-reply}
    total:=make([]uint64,k)
    for _,h:=range partial {for v,c:=range h {total[v]+=c}}
    return restore(total,lo)
}
func cold(a []int32,lo int32,k,p int) []int32 {
    pool:=newPool(p);out:=parallel(a,lo,k,p,pool);pool.close();return out
}
func equal(a,b []int32) bool {
    if len(a)!=len(b){return false};for i:=range a {if a[i]!=b[i]{return false}};return true
}
func selfTest(){
    pool:=newPool(3);defer pool.close();checks:=0
    for _,n:=range []int{0,1,2,7,100} {for _,k:=range []int{1,7,256} {
        lo:=int32(-k/2);a:=make([]int32,n)
        for i:=range a {a[i]=lo+int32((i*37+11)%k)}
        for shape:=0;shape<4;shape++ {
            expected:=append([]int32{},a...)
            sort.Slice(expected,func(i,j int)bool{return expected[i]<expected[j]})
            if !equal(sequential(a,lo,k),expected)||!equal(parallel(a,lo,k,3,pool),expected)||!equal(cold(a,lo,k,3),expected){panic("Self-test failed")}
            checks++
            if shape==0 {sort.Slice(a,func(i,j int)bool{return a[i]<a[j]})}
            if shape==1 {for i,j:=0,len(a)-1;i<j;i,j=i+1,j-1 {a[i],a[j]=a[j],a[i]}}
            if shape==2 {for i:=range a {a[i]=lo}}
        }
    }}
    fmt.Printf("Go PASS: %d datasets x 3 modes\n",checks)
}
func integer(s string) int {x,e:=strconv.Atoi(s);if e!=nil{panic(e)};return x}
type Config struct {mode string;index int}
func main(){
    if len(os.Args)==2 && os.Args[1]=="--self-test" {selfTest();return}
    if len(os.Args)!=7 {panic("Usage: INPUT LO K REPEATS OUTPUT_CSV P_LIST")}
    bytes,err:=os.ReadFile(os.Args[1]);if err!=nil {panic(err)}
    if len(bytes)%4!=0 {panic("Invalid i32 file")}
    a:=make([]int32,len(bytes)/4)
    for i:=range a {a[i]=int32(binary.LittleEndian.Uint32(bytes[i*4:i*4+4]))}
    lo,k,repeats:=int32(integer(os.Args[2])),integer(os.Args[3]),integer(os.Args[4])
    ps:=[]int{};for _,s:=range strings.Split(os.Args[6],","){ps=append(ps,integer(s))}
    pools:=[]*Pool{};for _,p:=range ps {pool:=newPool(p);pools=append(pools,pool);defer pool.close()}
    expected:=append([]int32{},a...);sort.Slice(expected,func(i,j int)bool{return expected[i]<expected[j]})
    configs:=[]Config{{"seq",0}};for i:=range ps {configs=append(configs,Config{"warm",i},Config{"cold",i})}
    run:=func(c Config) []int32 {
        if c.mode=="seq" {return sequential(a,lo,k)}
        if c.mode=="warm" {return parallel(a,lo,k,ps[c.index],pools[c.index])}
        return cold(a,lo,k,ps[c.index])
    }
    for _,c:=range configs {if !equal(run(c),expected){panic("Warm-up mismatch")}}
    file,err:=os.Create(os.Args[5]);if err!=nil {panic(err)};defer file.Close()
    fmt.Fprintln(file,"language,mode,p,repeat,ms")
    rng:=rand.New(rand.NewSource(42))
    for rep:=1;rep<=repeats;rep++ {
        rng.Shuffle(len(configs),func(i,j int){configs[i],configs[j]=configs[j],configs[i]})
        for _,c:=range configs {
            start:=time.Now();out:=run(c);ms:=float64(time.Since(start).Nanoseconds())/1e6
            if !equal(out,expected){panic("Result mismatch")}
            p:=1;if c.mode!="seq"{p=ps[c.index]}
            fmt.Fprintf(file,"Go,%s,%d,%d,%.12f\n",c.mode,p,rep,ms)
        }
    }
}
