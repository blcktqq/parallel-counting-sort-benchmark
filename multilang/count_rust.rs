use std::sync::{mpsc,Arc,Barrier,Mutex};
use std::thread::{self,JoinHandle};
use std::io::Write;
use std::time::Instant;
type Hist=Vec<u64>;
fn histogram(a:&[i32],lo:i32,k:usize)->Hist{
    let mut h=vec![0u64;k];
    for &x in a {
        let j=x as i64-lo as i64;
        assert!(j>=0 && (j as usize)<k,"Key out of range");
        h[j as usize]+=1;
    }
    h
}
fn restore(h:&[u64],lo:i32)->Vec<i32>{
    let n=h.iter().sum::<u64>() as usize;
    let mut out=vec![0i32;n];let mut pos=0;
    for (v,&count) in h.iter().enumerate(){
        let stop=pos+count as usize;out[pos..stop].fill(lo+v as i32);pos=stop;
    }
    out
}
fn sequential(a:&[i32],lo:i32,k:usize)->Vec<i32>{restore(&histogram(a,lo,k),lo)}
struct Job{block:Vec<i32>,lo:i32,k:usize,reply:mpsc::Sender<Hist>}
struct Pool{tx:mpsc::Sender<Option<Job>>,workers:Vec<JoinHandle<()>>}
impl Pool{
    fn new(p:usize)->Self{
        let (tx,rx)=mpsc::channel::<Option<Job>>();
        let rx=Arc::new(Mutex::new(rx));let ready=Arc::new(Barrier::new(p+1));
        let mut workers=Vec::new();
        for _ in 0..p{
            let rx=rx.clone();let ready=ready.clone();
            workers.push(thread::spawn(move||{
                ready.wait();
                loop{
                    let item={rx.lock().unwrap().recv().unwrap()};
                    match item{
                        Some(job)=>{let h=histogram(&job.block,job.lo,job.k);job.reply.send(h).unwrap();},
                        None=>break,
                    }
                }
            }));
        }
        ready.wait();Self{tx,workers}
    }
}
impl Drop for Pool{
    fn drop(&mut self){
        for _ in &self.workers{self.tx.send(None).unwrap();}
        for worker in self.workers.drain(..){worker.join().unwrap();}
    }
}
fn parallel(a:&[i32],lo:i32,k:usize,p:usize,pool:&Pool)->Vec<i32>{
    let blocks:Vec<Vec<i32>>=(0..p).map(|t|a[a.len()*t/p..a.len()*(t+1)/p].to_vec()).collect();
    let (tx,rx)=mpsc::channel();
    for block in blocks{pool.tx.send(Some(Job{block,lo,k,reply:tx.clone()})).unwrap();}
    drop(tx);
    let partial:Vec<Hist>=(0..p).map(|_|rx.recv().unwrap()).collect();
    let mut total=vec![0u64;k];
    for h in partial{for v in 0..k{total[v]+=h[v];}}
    restore(&total,lo)
}
fn cold(a:&[i32],lo:i32,k:usize,p:usize)->Vec<i32>{
    let pool=Pool::new(p);let out=parallel(a,lo,k,p,&pool);drop(pool);out
}
fn self_test(){
    let pool=Pool::new(3);let mut checks=0;
    for n in [0usize,1,2,7,100]{for k in [1usize,7,256]{
        let lo=-(k as i32)/2;let mut a:Vec<i32>=(0..n).map(|i|lo+((i*37+11)%k) as i32).collect();
        for shape in 0..4{
            let mut expected=a.clone();expected.sort_unstable();
            assert_eq!(sequential(&a,lo,k),expected);
            assert_eq!(parallel(&a,lo,k,3,&pool),expected);
            assert_eq!(cold(&a,lo,k,3),expected);checks+=1;
            match shape{0=>a.sort_unstable(),1=>a.reverse(),2=>a.fill(lo),_=>()}
        }
    }}
    println!("Rust PASS: {} datasets x 3 modes",checks);
}
fn main(){
    let args:Vec<String>=std::env::args().collect();
    if args.len()==2 && args[1]=="--self-test"{self_test();return;}
    assert_eq!(args.len(),7,"Usage: INPUT LO K REPEATS OUTPUT_CSV P_LIST");
    let bytes=std::fs::read(&args[1]).unwrap();assert_eq!(bytes.len()%4,0);
    let a:Vec<i32>=bytes.chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
    let lo:i32=args[2].parse().unwrap();let k:usize=args[3].parse().unwrap();
    let repeats:usize=args[4].parse().unwrap();
    let ps:Vec<usize>=args[6].split(',').map(|s|s.parse().unwrap()).collect();
    let pools:Vec<Pool>=ps.iter().map(|&p|Pool::new(p)).collect();
    let mut expected=a.clone();expected.sort_unstable();
    let mut configs=vec![("seq",0usize)];
    for i in 0..ps.len(){configs.push(("warm",i));configs.push(("cold",i));}
    let run=|mode:&str,i:usize|match mode{
        "seq"=>sequential(&a,lo,k),"warm"=>parallel(&a,lo,k,ps[i],&pools[i]),
        _=>cold(&a,lo,k,ps[i])};
    for &(mode,i) in &configs{assert_eq!(run(mode,i),expected);}
    let mut file=std::fs::File::create(&args[5]).unwrap();
    writeln!(file,"language,mode,p,repeat,ms").unwrap();
    let mut seed=42u64;
    for rep in 1..=repeats{
        for j in (1..configs.len()).rev(){
            seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x=(seed as usize)%(j+1);configs.swap(j,x);
        }
        for &(mode,i) in &configs{
            let start=Instant::now();let out=run(mode,i);let ms=start.elapsed().as_secs_f64()*1000.;
            assert_eq!(out,expected);
            writeln!(file,"Rust,{},{},{},{:.12}",mode,if mode=="seq"{1}else{ps[i]},rep,ms).unwrap();
        }
    }
}
