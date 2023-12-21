/*
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef MULTITHREADING_HPP_
#define MULTITHREADING_HPP_

#include <array>
#include <memory>
#include <thread>
#include <mutex>
#include <future>
#include <condition_variable>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace culib{
namespace multithreading{

namespace detail{

#ifdef __cpp_lib_hardware_interference_size
    inline constexpr std::size_t hardware_destructive_interference_size = std::hardware_destructive_interference_size;
#else
    inline constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

template<typename SizeT>
inline auto index_(SizeT cnt, SizeT cap){return cnt%cap;}

template<typename T>
class element_
{
    using value_type = T;
public:
    //left buffer uninitialized
    element_(){}

    template<typename...Args>
    void emplace(Args&&...args){
        new(buffer) value_type{std::forward<Args>(args)...};
    }
    template<typename V>
    void move(V& v){
        v = std::move(get());
    }
    void destroy(){
        get().~value_type();
    }
    value_type& get(){
        return *std::launder(reinterpret_cast<value_type*>(buffer));
    }
    const value_type& get()const{
        return *std::launder(reinterpret_cast<const value_type*>(buffer));
    }
private:
    alignas(value_type) std::byte buffer[sizeof(value_type)];
};

template<typename T>
class element
{
public:
    using value_type = T;
    ~element()
    {
        clear();
    }
    element(){}
    element(const element& other)
    {
        init(other);
    }
    element(element&& other)
    {
        init(std::move(other));
    }
    element& operator=(const element& other){
        clear();
        init(other);
        return *this;
    }
    element& operator=(element&& other){
        clear();
        init(std::move(other));
        return *this;
    }
    element& operator=(value_type&& v){
        clear();
        element__.emplace(std::move(v));
        empty_ = false;
        return *this;
    }
    operator bool()const{return !empty_;}
    auto empty()const{return empty_;}
    auto& get()const{return element__.get();}
    auto& get(){return element__.get();}
private:
    void clear(){
        if (!empty_){
            element__.destroy();
        }
    }
    void init(const element& other){
        if (!other.empty_){
            element__.emplace(other.element__.get());
            empty_ = false;
        }
    }
    void init(element&& other){
        if (!other.empty_){
            element__.emplace(std::move(other.element__.get()));
            empty_ = false;
        }
    }

    element_<value_type> element__{};
    bool empty_{true};
};

//thread_pool_v3 helpers
//syncronize completion of group of tasks
class task_group
{
    std::atomic<std::size_t> task_inprogress_counter_{0};
    std::condition_variable all_complete_{};
    std::mutex guard_{};
public:
    void inc(){
        ++task_inprogress_counter_;
    }
    void dec(){
        std::lock_guard<std::mutex> lock{guard_};
        --task_inprogress_counter_;
        all_complete_.notify_all();
    }
    void wait(){
        std::unique_lock<std::mutex> lock{guard_};
        while(task_inprogress_counter_.load()!=0){
            all_complete_.wait(lock);
        }
    }
};

template<typename R>
class task_future{
    using result_type = R;
    bool sync_;
    std::future<result_type> f;
public:
    ~task_future(){
        if (sync_ && f.valid()){
            f.wait();
        }
    }
    task_future() = default;
    task_future(task_future&&) = default;
    task_future& operator=(task_future&&) = default;
    task_future(bool sync__, std::future<result_type>&& f_):
        sync_{sync__},
        f{std::move(f_)}
    {}
    operator bool(){return f.valid();}
    void wait()const{f.wait();}
    auto get(){return f.get();}
};

class task_v3_base
{
public:
    virtual ~task_v3_base(){}
    virtual void call() = 0;
};

template<typename F, typename...Args>
class task_v3_impl : public task_v3_base
{
    using args_type = decltype(std::make_tuple(std::declval<Args>()...));
    using result_type = std::decay_t<decltype(std::apply(std::declval<F>(),std::declval<args_type>()))>;
    F f;
    args_type args;
    std::promise<result_type> task_promise;
    void call() override {
            if constexpr(std::is_void_v<result_type>){
                std::apply(f, std::move(args));
                task_promise.set_value();
            }else{
                task_promise.set_value(std::apply(f, std::move(args)));
            }
        }
public:
    using future_type = task_future<result_type>;
    template<typename F_, typename...Args_>
    task_v3_impl(F_&& f_, Args_&&...args_):
            f{std::forward<F_>(f_)},
            args{std::make_tuple(std::forward<Args_>(args_)...)}
        {}
    auto get_future(bool sync = true){
            return future_type{sync, task_promise.get_future()};
        }
};

template<typename F, typename...Args>
class group_task_v3_impl : public task_v3_base
{
    using args_type = decltype(std::make_tuple(std::declval<Args>()...));
    std::reference_wrapper<task_group> group_;
    F f;
    args_type args;
    void call() override {
        std::apply(f, std::move(args));
        group_.get().dec();
    }
public:
    template<typename F_, typename...Args_>
    group_task_v3_impl(std::reference_wrapper<task_group> group__, F_&& f_, Args_&&...args_):
        group_{group__},
        f{std::forward<F_>(f_)},
        args{std::make_tuple(std::forward<Args_>(args_)...)}
    {}
};

class task_v3
{
    std::unique_ptr<task_v3_base> impl;
public:
    task_v3() = default;
    void call(){
        impl->call();
    }
    template<typename F, typename...Args>
    auto set_task(bool sync, F&& f, Args&&...args){
        using impl_type = task_v3_impl<std::decay_t<F>, std::decay_t<Args>...>;
        impl = std::make_unique<impl_type>(std::forward<F>(f),std::forward<Args>(args)...);
        return static_cast<impl_type*>(impl.get())->get_future(sync);
    }
    template<typename F, typename...Args>
    void set_group_task(std::reference_wrapper<task_group> group, F&& f, Args&&...args){
        using impl_type = group_task_v3_impl<std::decay_t<F>, std::decay_t<Args>...>;
        impl = std::make_unique<impl_type>(group, std::forward<F>(f),std::forward<Args>(args)...);
    }
};

}   //end of namespace detail

//multiple producer multiple consumer bounded queue
template<typename T, typename Allocator = std::allocator<detail::element_<T>>>
class mpmc_bounded_queue_v3
{
    using element_type = typename std::allocator_traits<Allocator>::value_type;
    using size_type = std::size_t;
    using mutex_type = std::mutex;
    static_assert(std::is_unsigned_v<size_type>);
public:
    using value_type = T;
    using allocator_type = Allocator;

    mpmc_bounded_queue_v3(size_type capacity__, const allocator_type& alloc = allocator_type()):
        capacity_{capacity__},
        allocator{alloc}
    {
        if (capacity_ == 0){
            throw std::invalid_argument("queue capacity must be > 0");
        }
        elements = allocator.allocate(capacity_+1);
    }
    ~mpmc_bounded_queue_v3()
    {
        clear();
        allocator.deallocate(elements, capacity_+1);
    }

    template<typename...Args>
    bool try_push(Args&&...args){
        std::unique_lock<mutex_type> lock{push_guard};
        auto push_index_ = push_index.load(std::memory_order::memory_order_relaxed);
        auto next_push_index = index(push_index_+1);
        if (next_push_index == pop_index.load(std::memory_order::memory_order_acquire)){//queue is full
            lock.unlock();
            return false;
        }else{
            elements[push_index_].emplace(std::forward<Args>(args)...);
            push_index.store(next_push_index, std::memory_order::memory_order_release);
            lock.unlock();
            return true;
        }
    }

    bool try_pop(value_type& v){
        return try_pop_(v);
    }
    auto try_pop(){
        detail::element<value_type> v{};
        try_pop_(v);
        return v;
    }

    template<typename...Args>
    void push(Args&&...args){
        std::unique_lock<mutex_type> lock{push_guard};
        auto push_index_ = push_index.load(std::memory_order::memory_order_relaxed);
        auto next_push_index = index(push_index_+1);
        while(next_push_index == pop_index.load(std::memory_order::memory_order_acquire));//wait until not full
        elements[push_index_].emplace(std::forward<Args>(args)...);
        push_index.store(next_push_index, std::memory_order::memory_order_release);
        lock.unlock();
    }

    void pop(value_type& v){
        pop_(v);
    }
    auto pop(){
        detail::element<value_type> v{};
        pop_(v);
        return v;
    }

    auto size()const{
        auto push_index_ = push_index.load(std::memory_order::memory_order_relaxed);
        auto pop_index_ = pop_index.load(std::memory_order::memory_order_relaxed);
        return pop_index_ > push_index_ ? (capacity_+1+push_index_-pop_index_) : (push_index_ - pop_index_);
    }
    auto capacity()const{return capacity_;}

private:

    template<typename V>
    bool try_pop_(V& v){
        std::unique_lock<mutex_type> lock{pop_guard};
        auto pop_index_ = pop_index.load(std::memory_order::memory_order_relaxed);
        if (pop_index_ == push_index.load(std::memory_order::memory_order_acquire)){//queue is empty
            lock.unlock();
            return false;
        }else{
            elements[pop_index].move(v);
            elements[pop_index].destroy();
            pop_index.store(index(pop_index_+1), std::memory_order::memory_order_release);
            lock.unlock();
            return true;
        }
    }

    template<typename V>
    void pop_(V& v){
        std::unique_lock<mutex_type> lock{pop_guard};
        auto pop_index_ = pop_index.load(std::memory_order::memory_order_relaxed);
        while(pop_index_ == push_index.load(std::memory_order::memory_order_acquire));//wait until not empty
        elements[pop_index].move(v);
        elements[pop_index].destroy();
        pop_index.store(index(pop_index_+1), std::memory_order::memory_order_release);
        lock.unlock();
    }

    void clear(){
        auto pop_index_ = pop_index.load(std::memory_order::memory_order_relaxed);
        auto push_index_ = push_index.load(std::memory_order::memory_order_relaxed);
        while(pop_index_ != push_index_){
            elements[pop_index].destroy();
            pop_index_ = index(pop_index_+1);
        }
    }

    auto index(size_type cnt){return detail::index_(cnt, capacity_+1);}

    size_type capacity_;
    allocator_type allocator;
    element_type* elements{};
    std::atomic<size_type> push_index{};
    std::atomic<size_type> pop_index{};
    mutex_type push_guard{};
    std::array<std::byte, detail::hardware_destructive_interference_size> padding_;
    mutex_type pop_guard{};
};

//single thread bounded queue
template<typename T, typename Allocator = std::allocator<detail::element_<T>>>
class st_bounded_queue
{
    using element_type = typename std::allocator_traits<Allocator>::value_type;
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = typename std::allocator_traits<Allocator>::size_type;

    st_bounded_queue(size_type capacity__, const allocator_type& alloc = allocator_type()):
        capacity_{capacity__},
        allocator{alloc}
    {
        if (capacity_ == 0){
            throw std::invalid_argument("queue capacity must be > 0");
        }
        elements = allocator.allocate(capacity_+1);
    }
    ~st_bounded_queue()
    {
        clear();
        allocator.deallocate(elements, capacity_+1);
    }

    template<typename...Args>
    auto try_push(Args&&...args){
        value_type* res{nullptr};
        auto next_push_index = index(push_index+1);
        if (next_push_index==pop_index){
            return res;
        }else{
            elements[push_index].emplace(std::forward<Args>(args)...);
            res = &elements[push_index].get();
            push_index = next_push_index;
            return res;
        }
    }

    bool try_pop(value_type& v){
        return try_pop_(&v);
    }
    auto try_pop(){
        detail::element<value_type> v{};
        try_pop_(&v);
        return v;
    }
    bool pop(){
        return try_pop_(nullptr);
    }

    value_type* front(){return front_helper();}
    const value_type* front()const{return front_helper();}

    bool empty()const{return push_index == pop_index;}
    size_type size()const{return pop_index > push_index ? (capacity_+1+push_index-pop_index) : (push_index - pop_index);}
    size_type capacity()const{return capacity_;}

private:

    value_type* front_helper()const{
        return empty() ? nullptr : &elements[pop_index].get();
    }

    //call with nullptr arg will destroy front and update pop_index
    //call with valid pointer additionaly move element to v
    //in any case return true if queue no empty false otherwise
    template<typename V>
    bool try_pop_(V* v){
        if (empty()){
            return false;
        }else{
            if (v){
                elements[pop_index].move(*v);   //move assign from elements to v
            }
            elements[pop_index].destroy();
            pop_index = index(pop_index+1);
            return true;
        }
    }

    void clear(){
        while(!empty()){
            elements[pop_index].destroy();
            pop_index = index(pop_index+1);
        }
    }

    auto index(size_type cnt){return cnt%(capacity_+1);}

    size_type capacity_;
    allocator_type allocator;
    element_type* elements;
    size_type push_index{0};
    size_type pop_index{0};
};

using detail::task_group;

//single allocation thread pool with bounded task queue
//allow different signatures and return types of task callable
//push task template method returns task_future<R>, where R is return type of callable given arguments types
class thread_pool_v3
{
    using task_type = detail::task_v3;
    using queue_type = st_bounded_queue<task_type>;
    using mutex_type = std::mutex;

public:

    ~thread_pool_v3()
    {
        stop();
    }
    thread_pool_v3(std::size_t n_workers):
        thread_pool_v3(n_workers, n_workers)
    {}
    thread_pool_v3(std::size_t n_workers, std::size_t n_tasks):
        workers(n_workers),
        tasks(n_tasks)
    {
        init();
    }

    //return task_future<R> object, where R is return type of F called with args, future will sync when destroyed
    //std::reference_wrapper should be used to pass args by ref
    template<typename F, typename...Args>
    auto push(F&& f, Args&&...args){return push_<true>(std::forward<F>(f), std::forward<Args>(args)...);}
    //returned future will not sync when destroyed
    template<typename F, typename...Args>
    auto push_async(F&& f, Args&&...args){return push_<false>(std::forward<F>(f), std::forward<Args>(args)...);}
    //bind task to group
    template<typename F, typename...Args>
    void push_group(task_group& group, F&& f, Args&&...args){
        std::unique_lock<mutex_type> lock{guard};
        while(true){
            if (auto task = tasks.try_push()){
                group.inc();
                task->set_group_task(group, std::forward<F>(f), std::forward<Args>(args)...);
                has_task.notify_one();
                lock.unlock();
                break;
            }else{
                has_slot.wait(lock);
            }
        }
    }

private:

    template<bool Sync = true, typename F, typename...Args>
    auto push_(F&& f, Args&&...args){
        using future_type = decltype( std::declval<task_type>().set_task(Sync, std::forward<F>(f), std::forward<Args>(args)...));
        std::unique_lock<mutex_type> lock{guard};
        while(true){
            if (auto task = tasks.try_push()){
                future_type future = task->set_task(Sync, std::forward<F>(f), std::forward<Args>(args)...);
                has_task.notify_one();
                lock.unlock();
                return future;
            }else{
                has_slot.wait(lock);
            }
        }
    }

    void init(){
        std::for_each(workers.begin(),workers.end(),[this](auto& worker){worker=std::thread{&thread_pool_v3::worker_loop, this};});
    }

    void stop(){
        std::unique_lock<mutex_type> lock{guard};
        finish_workers.store(true);
        has_task.notify_all();
        lock.unlock();
        std::for_each(workers.begin(),workers.end(),[](auto& worker){worker.join();});
    }

    //problem is to use waiting not yealding in loop and have concurrent push and pop
    //conditional_variable must use same mutex to guard push and pop, even if queue is mpmc
    void worker_loop(){
        while(!finish_workers.load()){  //worker loop
            std::unique_lock<mutex_type> lock{guard};
            while(!finish_workers.load()){  //has_task conditional loop
                if (auto t = tasks.try_pop()){
                    has_slot.notify_one();
                    lock.unlock();
                    t.get().call();
                    break;
                }else{
                    has_task.wait(lock);
                }
            }
        }
    }

    std::vector<std::thread> workers;
    queue_type tasks;
    std::atomic<bool> finish_workers{false};
    mutex_type guard;
    std::condition_variable has_task;
    std::condition_variable has_slot;
};

inline constexpr std::size_t pool_workers_n = 4;
inline constexpr std::size_t pool_queue_size = 16;
inline auto& get_pool(){
    static thread_pool_v3 pool_{pool_workers_n, pool_queue_size};
    return pool_;
}

namespace detail{

//mc_bounded_pool helpers
class queue_of_refs
{
    using queue_type = mpmc_bounded_queue_v3<void*>;
    queue_type refs;
public:
    queue_of_refs(std::size_t capacity_):
        refs{capacity_}
    {}
    auto size()const{return refs.size();}
    auto capacity()const{return refs.capacity();}
    void push(void* ref){refs.push(ref);}
    auto pop(){return refs.pop();}
    auto try_pop(){return refs.try_pop();}
};

template<typename T>
class shareable_element{
    using value_type = T;
    using pool_type = queue_of_refs;
public:
    template<typename...Args>
    shareable_element(pool_type* pool_, Args&&...args):
        pool{pool_},
        value{std::forward<Args>(args)...}
    {}
    auto make_shared(){
        inc_ref();
        return shared_element{this};
    }
    static auto make_empty_shared(){
        return shared_element{nullptr};
    }
private:
    pool_type* pool;
    value_type value;
    std::atomic<std::size_t> use_count_{0};

    class shared_element{
        friend class shareable_element;
        shareable_element* elem;
        shared_element(shareable_element* elem_):
            elem{elem_}
        {}
        void inc_ref(){
            if (elem){
                elem->inc_ref();
            }
        }
        void dec_ref(){
            if (elem){
                elem->dec_ref();
            }
        }
    public:
        ~shared_element()
        {
            dec_ref();
        }
        shared_element() = default;
        shared_element(const shared_element& other):
            elem{other.elem}
        {
            inc_ref();
        }
        shared_element(shared_element&& other):
            elem{other.elem}
        {
            other.elem = nullptr;
        }
        shared_element& operator=(const shared_element& other){
            dec_ref();
            elem = other.elem;
            inc_ref();
            return *this;
        }
        shared_element& operator=(shared_element&& other){
            dec_ref();
            elem = other.elem;
            other.elem = nullptr;
            return *this;
        }
        operator bool()const{return static_cast<bool>(elem);}
        void reset(){
            dec_ref();
            elem = nullptr;
        }
        auto use_count()const{return elem->use_count();}
        auto& get(){return elem->get();}
        auto& get()const{return elem->get();}
    };

    auto inc_ref(){return use_count_.fetch_add(1);}
    auto dec_ref(){
        if (use_count_.fetch_sub(1) == 1){
            pool->push(this);
        }
    }
    auto use_count()const{return use_count_.load();}
    auto& get(){return value;}
    auto& get()const{return value;}
};

}   //end of namespace detail

//multiple consumer bounded pool of reusable objects
template<typename T, typename Allocator = std::allocator<detail::shareable_element<T>>>
class mc_bounded_pool
{

    using element_type = typename std::allocator_traits<Allocator>::value_type;
    using pool_type = detail::queue_of_refs;

public:
    using value_type = T;
    using allocator_type = Allocator;

    template<typename...Args>
    explicit mc_bounded_pool(std::size_t capacity__, Args&&...args):
        allocator{allocator_type{}},
        pool(capacity__),
        elements{allocator.allocate(capacity__)}
    {
        init(std::forward<Args>(args)...);
    }

    template<typename It, std::enable_if_t<!std::is_convertible_v<It,std::size_t>,int> = 0>
    mc_bounded_pool(It first, It last, const Allocator& allocator__ = Allocator{}):
        allocator{allocator__},
        pool(std::distance(first,last)),
        elements{allocator.allocate(pool.capacity())}
    {
        init(first, last);
    }

    ~mc_bounded_pool()
    {
        clear();
        allocator.deallocate(elements,capacity());
    }

    //returns shared_element object that is smart wrapper of reference to reusable object
    //after last copy of shared_element object is destroyed reusable object is returned to pool and ready to new use
    //blocks until reusable object is available
    auto pop(){
        return static_cast<element_type*>(pool.pop().get())->make_shared();
    }

    //not blocking, if no reusable objects available returns immediately
    //result, that is shared_element object, can be converted to bool and test, false if no objects available, true otherwise
    auto try_pop(){
        if (auto e = pool.try_pop()){
            return static_cast<element_type*>(e.get())->make_shared();
        }else{
            return element_type::make_empty_shared();
        }
    }

    auto size()const{return pool.size();}
    auto capacity()const{return pool.capacity();}
    auto empty()const{return size() == 0;}
private:
    template<typename...Args>
    void init(Args&&...args){
        auto it = elements;
        auto end = elements+capacity();
        for(;it!=end;++it){new(it) element_type{&pool, std::forward<Args>(args)...};}
        init_pool();
    }

    template<typename It>
    void init(It first, It last){
        auto it = elements;
        for(;first!=last;++it,++first){new(it) element_type{&pool, *first};}
        init_pool();
    }
    void init_pool(){std::for_each(elements,elements+capacity(),[this](auto& e){pool.push(&e);});}
    void clear(){std::destroy(elements,elements+capacity());}

    allocator_type allocator;
    pool_type pool;
    element_type* elements;
};

}   //end of namespace multithreading
}   //end of namespace culib
#endif
